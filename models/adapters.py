"""Zero-initialized residual adapters used by the CARL-D detector.

The adapters in this module are deliberately detector-shaped: convolutional
adapters operate on feature maps and the linear adapter operates on the RoI
representation.  The final projection is initialized to zero, so inserting an
adapter is an exact no-op until it is trained.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


def adapter_hidden_size(width: int, reduction: int) -> int:
    """Return the bottleneck width shared by parameter estimates and modules."""
    if width <= 0:
        raise ValueError("width must be positive")
    if reduction <= 0:
        raise ValueError("reduction must be positive")
    return max(1, width // reduction)


def residual_adapter_parameter_count(width: int, reduction: int) -> int:
    """Return the exact number of parameters in one bias-free adapter."""
    hidden = adapter_hidden_size(width, reduction)
    return 2 * width * hidden


class ResidualConvAdapter(nn.Module):
    """A bottleneck residual adapter for a ``[N, C, H, W]`` feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = adapter_hidden_size(channels, reduction)
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=False)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return residual_adapter_parameter_count(self.channels, self.reduction)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        """Return only the learned residual branch."""
        return self.up(self.activation(self.down(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.residual(features)


class ResidualLinearAdapter(nn.Module):
    """A bottleneck residual adapter for the flattened RoI representation."""

    def __init__(self, features: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = adapter_hidden_size(features, reduction)
        self.features = int(features)
        self.reduction = int(reduction)
        self.down = nn.Linear(features, hidden, bias=False)
        self.activation = nn.ReLU(inplace=False)
        self.up = nn.Linear(hidden, features, bias=False)

        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return residual_adapter_parameter_count(self.features, self.reduction)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        """Return only the learned residual branch."""
        return self.up(self.activation(self.down(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.residual(features)


class FeatureMapAdapterBank(nn.Module):
    """One expansion generation over an ordered detector feature pyramid."""

    def __init__(
        self,
        feature_names: Sequence[str],
        channels: Sequence[int],
        reduction: int = 16,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in feature_names)
        widths = tuple(int(width) for width in channels)
        if not names or len(names) != len(widths):
            raise ValueError(
                "feature_names and channels must have the same non-zero length"
            )
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if any("." in name for name in names):
            raise ValueError(
                "feature names containing '.' cannot be registered by ModuleDict"
            )

        self.feature_names = names
        self.channels = widths
        self.reduction = int(reduction)
        self.adapters = nn.ModuleDict(
            {
                name: ResidualConvAdapter(width, reduction=self.reduction)
                for name, width in zip(names, widths)
            }
        )

    @property
    def parameter_count(self) -> int:
        return sum(
            residual_adapter_parameter_count(width, self.reduction)
            for width in self.channels
        )

    def forward(
        self, features: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        actual_names = tuple(str(name) for name in features)
        if actual_names != self.feature_names:
            raise ValueError(
                "feature pyramid changed after adapter construction: "
                f"expected {self.feature_names}, got {actual_names}"
            )
        return OrderedDict(
            (name, self.adapters[name](features[name])) for name in self.feature_names
        )

    def residual(
        self, features: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Return only this generation's residual feature maps."""
        actual_names = tuple(str(name) for name in features)
        if actual_names != self.feature_names:
            raise ValueError(
                "feature pyramid changed after adapter construction: "
                f"expected {self.feature_names}, got {actual_names}"
            )
        return OrderedDict(
            (name, self.adapters[name].residual(features[name]))
            for name in self.feature_names
        )


def _validate_merge_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode != "parallel":
        raise ValueError("CARL uses parallel adapters; sequential checkpoints are unsupported")
    return mode


def _merge_feature_adapters(
    features: OrderedDict[str, torch.Tensor],
    adapters: nn.ModuleList,
    mode: str,
) -> OrderedDict[str, torch.Tensor]:
    _validate_merge_mode(mode)
    if not adapters:
        return features
    base = features
    merged = OrderedDict((name, value) for name, value in base.items())
    for adapter in adapters:
        residual = adapter.residual(base)
        merged = OrderedDict((name, merged[name] + residual[name]) for name in base)
    return merged


class ExpandableBackboneWithFPN(nn.Module):
    """Wrap Torchvision's BackboneWithFPN with independent adapter groups."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_feature_names: Sequence[str],
        backbone_channels: Sequence[int],
        fpn_feature_names: Sequence[str],
        adapter_merge: str = "parallel",
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "body") or not hasattr(backbone, "fpn"):
            raise TypeError("backbone must expose body and fpn modules")
        if not hasattr(backbone, "out_channels"):
            raise TypeError("backbone must expose out_channels")

        self.body = backbone.body
        self.fpn = backbone.fpn
        self.out_channels = int(backbone.out_channels)
        self.backbone_feature_names = tuple(
            str(name) for name in backbone_feature_names
        )
        self.backbone_channels = tuple(int(width) for width in backbone_channels)
        self.fpn_feature_names = tuple(str(name) for name in fpn_feature_names)
        self.adapter_merge = _validate_merge_mode(adapter_merge)
        if len(self.backbone_feature_names) != len(self.backbone_channels):
            raise ValueError(
                "backbone feature names and channels must have equal length"
            )

        self.backbone_adapters = nn.ModuleList()
        self.fpn_adapters = nn.ModuleList()

    def estimate_expansion_params(self, group: str, reduction: int = 16) -> int:
        if group == "backbone":
            widths = self.backbone_channels
        elif group == "fpn":
            widths = (self.out_channels,) * len(self.fpn_feature_names)
        else:
            raise ValueError(f"unsupported backbone adapter group: {group}")
        return sum(
            residual_adapter_parameter_count(width, reduction) for width in widths
        )

    def add_adapter(self, group: str, reduction: int = 16) -> nn.Module:
        if group == "backbone":
            bank = FeatureMapAdapterBank(
                self.backbone_feature_names,
                self.backbone_channels,
                reduction,
            )
            self.backbone_adapters.append(bank)
        elif group == "fpn":
            bank = FeatureMapAdapterBank(
                self.fpn_feature_names,
                (self.out_channels,) * len(self.fpn_feature_names),
                reduction,
            )
            self.fpn_adapters.append(bank)
        else:
            raise ValueError(f"unsupported backbone adapter group: {group}")
        return bank

    def set_adapter_merge(self, mode: str) -> None:
        self.adapter_merge = _validate_merge_mode(mode)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        # ``Module.compile`` installs a compiled ``__call__`` while leaving the
        # original ``forward`` available. Use the compiled path only for
        # training. Controller probes and replay refinement run under
        # inference/no-grad modes and with many remainder batch sizes; sending
        # those calls through the training compiler cache creates unnecessary
        # grad-mode/shape variants and can exhaust Dynamo's recompile limit.
        features = self.body(images) if self.training else self.body.forward(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict((("0", features),))
        features = _merge_feature_adapters(
            features, self.backbone_adapters, self.adapter_merge
        )

        features = self.fpn(features) if self.training else self.fpn.forward(features)
        features = _merge_feature_adapters(
            features, self.fpn_adapters, self.adapter_merge
        )
        return features


class ExpandableRoIBoxHead(nn.Module):
    """Wrap a Torchvision RoI box head with residual expansion generations."""

    def __init__(
        self,
        box_head: nn.Module,
        output_features: int,
        adapter_merge: str = "parallel",
    ) -> None:
        super().__init__()
        self.base_head = box_head
        self.output_features = int(output_features)
        self.adapter_merge = _validate_merge_mode(adapter_merge)
        self.adapters = nn.ModuleList()

    def estimate_expansion_params(self, reduction: int = 16) -> int:
        return residual_adapter_parameter_count(self.output_features, reduction)

    def add_adapter(self, reduction: int = 16) -> ResidualLinearAdapter:
        adapter = ResidualLinearAdapter(self.output_features, reduction)
        self.adapters.append(adapter)
        return adapter

    def set_adapter_merge(self, mode: str) -> None:
        self.adapter_merge = _validate_merge_mode(mode)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.base_head(features)
        if not self.adapters:
            return features
        return features + sum(
            (adapter.residual(features) for adapter in self.adapters), start=0
        )
