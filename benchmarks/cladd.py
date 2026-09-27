"""Detection-only CLAD-D benchmark integration.

The official CLAD repository owns the dataset split, target construction, and
box-aware transforms.  This module provides the small amount of CARL-D glue
needed around it: deterministic evaluation transforms, four domain-specific
loaders, and stable domain metadata.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_CLAD_REPO_ROOT_ENV = "CARL_CLAD_REPO_ROOT"
_CLAD_SUPPRESS_ENV = "CLAD_SUPPRESS_DETECTRON_MSG"

NUM_TASKS = 4
NUM_FOREGROUND_CLASSES = 6
NUM_DETECTOR_CLASSES = NUM_FOREGROUND_CLASSES + 1
BACKGROUND_CLASS_ID = 0

CLASS_NAMES: Mapping[int, str] = {
    1: "Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle",
}

DOMAIN_NAMES: Tuple[str, ...] = (
    "clear_day_citystreet",
    "day_highway",
    "night",
    "rainy_day",
)

# These definitions mirror clad.utils.meta.  Train/validation task 1 is the
# Shanghai subset, while its official test domain covers both city streets and
# country roads.
TRAIN_DOMAIN_DEFINITIONS: Tuple[Mapping[str, Any], ...] = (
    {
        "city": "Shanghai",
        "location": "Citystreet",
        "period": "Daytime",
        "weather": "Clear",
    },
    {
        "location": "Highway",
        "period": "Daytime",
        "weather": ("Clear", "Overcast"),
    },
    {"period": "Night"},
    {"period": "Daytime", "weather": "Rainy"},
)

TEST_DOMAIN_DEFINITIONS: Tuple[Mapping[str, Any], ...] = (
    {
        "location": ("Citystreet", "Countryroad"),
        "period": "Daytime",
        "weather": ("Clear", "Overcast"),
    },
    {
        "location": "Highway",
        "period": "Daytime",
        "weather": ("Clear", "Overcast"),
    },
    {"period": "Night"},
    {"period": "Daytime", "weather": "Rainy"},
)


def _import_clad(clad_repo_root: str):
    """Import CLAD from the repository explicitly selected in the config."""
    if not clad_repo_root:
        raise ValueError("benchmark.clad_repo_root must not be empty")

    configured_root = os.path.realpath(os.path.expanduser(clad_repo_root))
    if not os.path.isdir(configured_root):
        raise FileNotFoundError(
            f"Configured CLAD repository does not exist: {configured_root}"
        )

    loaded_module = sys.modules.get("clad")
    if loaded_module is not None:
        module_file = getattr(loaded_module, "__file__", None)
        if module_file is not None:
            loaded_path = os.path.realpath(module_file)
            try:
                is_selected_checkout = (
                    os.path.commonpath((configured_root, loaded_path))
                    == configured_root
                )
            except ValueError:
                is_selected_checkout = False
            if not is_selected_checkout:
                raise RuntimeError(
                    f"clad is already imported from {loaded_path}, but the "
                    f"configured repository is {configured_root}"
                )
        return loaded_module

    if configured_root not in sys.path:
        sys.path.insert(0, configured_root)
    importlib.invalidate_caches()

    if os.environ.get(_CLAD_SUPPRESS_ENV, "0") == "1":
        with redirect_stdout(StringIO()):
            return importlib.import_module("clad")
    return importlib.import_module("clad")


def _preload_clad_for_spawned_worker() -> None:
    """Import CLAD before a spawned worker deserializes an official dataset."""
    clad_repo_root = os.environ.get(_CLAD_REPO_ROOT_ENV)
    if clad_repo_root:
        _import_clad(clad_repo_root)


_preload_clad_for_spawned_worker()


@dataclass(frozen=True)
class DetectionDatasetMetadata:
    """Stable metadata attached to one CLAD-D domain dataset."""

    domain_id: int
    domain_name: str
    split: str
    definition: Mapping[str, Any]
    class_names: Mapping[int, str]
    num_foreground_classes: int = NUM_FOREGROUND_CLASSES
    num_detector_classes: int = NUM_DETECTOR_CLASSES
    background_class_id: int = BACKGROUND_CLASS_ID


class DetectionDomainDataset(Dataset):
    """Attach stable domain metadata to an official CLAD-D dataset."""

    def __init__(
        self,
        official_dataset: Dataset,
        metadata: DetectionDatasetMetadata,
    ) -> None:
        self.official_dataset = official_dataset
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.official_dataset)

    def __getitem__(self, index: int):
        return self.official_dataset[index]

    @property
    def ids(self):
        """Expose official image IDs without treating object labels as samples."""
        return getattr(self.official_dataset, "ids", None)


def _require_four_domains(
    datasets: Sequence[Dataset],
    *,
    split: str,
) -> None:
    if len(datasets) != NUM_TASKS:
        raise ValueError(
            f"Official CLAD-D {split} loader returned {len(datasets)} "
            f"domains; expected exactly {NUM_TASKS}."
        )


class CLADDBenchmark:
    """Four-domain CLAD-D benchmark backed by the official CLAD loader."""

    def __init__(
        self,
        root_dir: str,
        clad_module,
        batch_size: int = 2,
        eval_batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        rare_image_sampling: bool = True,
        rare_image_sampling_strength: float = 0.5,
        class_balance_beta: float = 0.999,
        class_balance_max_weight: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.root_dir = root_dir
        self.clad = clad_module
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.rare_image_sampling = bool(rare_image_sampling)
        self.rare_image_sampling_strength = float(rare_image_sampling_strength)
        self.class_balance_beta = float(class_balance_beta)
        self.class_balance_max_weight = float(class_balance_max_weight)
        self.seed = seed

        required_api = (
            "get_cladd_trainval",
            "get_cladd_test",
            "get_transform",
            "collate_fn_cladd",
        )
        missing_api = [name for name in required_api if not hasattr(clad_module, name)]
        if missing_api:
            raise AttributeError(
                "Configured CLAD checkout is missing detection APIs: "
                + ", ".join(missing_api)
            )

        # Always pass transforms explicitly.  In the official loader the test
        # default is a training transform, so relying on that default would make
        # final evaluation randomly flip images.
        self.train_transform = self.clad.get_transform(train=True)
        self.eval_transform = self.clad.get_transform(train=False)
        train_steps = tuple(getattr(self.train_transform, "transforms", ()))
        eval_steps = tuple(getattr(self.eval_transform, "transforms", ()))
        if (
            not eval_steps
            or len(eval_steps) > len(train_steps)
            or tuple(type(step) for step in train_steps[: len(eval_steps)])
            != tuple(type(step) for step in eval_steps)
        ):
            raise RuntimeError(
                "Official CLAD-D train transforms must begin with the "
                "deterministic evaluation transform"
            )
        self._replay_train_suffix = train_steps[len(eval_steps) :]
        train_sets, val_sets = self.clad.get_cladd_trainval(
            self.root_dir,
            train_transform=self.train_transform,
            val_transform=self.eval_transform,
            avalanche=False,
        )
        _require_four_domains(train_sets, split="train")
        _require_four_domains(val_sets, split="validation")

        self._train_sets = self._wrap_domains(train_sets, "train")
        self._val_sets = self._wrap_domains(val_sets, "validation")
        self._test_sets: Optional[List[Dataset]] = None

    def prepare_replay_tensor(self, image: Any) -> torch.Tensor:
        """Apply the deterministic official transform once for replay caching."""
        tensor, _ = self.eval_transform(image, {})
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            raise TypeError(
                "Official CLAD-D evaluation transform must produce [C,H,W] tensors"
            )
        return tensor.detach().contiguous()

    def transform_cached_replay(
        self, image: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply the stochastic suffix and record whether it mirrored the image.

        The private marker is consumed by ``DetectionReplayBuffer`` before the
        target reaches Torchvision.  Distillation uses it to mirror cached,
        normalized teacher proposals with exactly the same augmentation.
        """
        if not torch.is_tensor(image) or image.ndim != 3:
            raise TypeError("Cached CLAD-D replay images must have shape [C,H,W]")
        horizontally_flipped = False
        for transform in self._replay_train_suffix:
            previous_image = image
            image, target = transform(image, target)
            # The official suffix is RandomHorizontalFlip: it returns the
            # input tensor unchanged on the identity path and a new flipped
            # tensor on the augmentation path.
            if image is not previous_image:
                horizontally_flipped = not horizontally_flipped
        target["_carl_replay_hflip"] = horizontally_flipped
        return image, target

    def _rare_image_weights(self, dataset: Dataset) -> torch.Tensor:
        """Return mild multi-label image weights from official annotations."""
        official = getattr(dataset, "official_dataset", dataset)
        ids = getattr(official, "ids", None)
        annotations = getattr(official, "img_anns", None)
        image_classes: list[tuple[int, ...]] = []
        object_counts = torch.zeros(NUM_DETECTOR_CLASSES, dtype=torch.float64)
        if ids is not None and annotations is not None:
            for image_id in ids:
                labels = tuple(
                    int(annotation["category_id"])
                    for annotation in annotations[image_id]
                )
                image_classes.append(tuple(sorted(set(labels))))
                if labels:
                    object_counts.add_(
                        torch.bincount(
                            torch.tensor(labels, dtype=torch.int64),
                            minlength=NUM_DETECTOR_CLASSES,
                        ).to(dtype=torch.float64)
                    )
        else:
            load_target = getattr(official, "_load_target", None)
            for index in range(len(dataset)):
                target = (
                    load_target(index) if callable(load_target) else official[index][1]
                )
                labels = torch.as_tensor(target["labels"], dtype=torch.int64)
                image_classes.append(tuple(sorted(set(labels.tolist()))))
                object_counts.add_(
                    torch.bincount(labels, minlength=NUM_DETECTOR_CLASSES).to(
                        dtype=torch.float64
                    )
                )

        beta = self.class_balance_beta
        foreground = object_counts[1:]
        effective = torch.ones_like(foreground)
        present = foreground > 0
        effective[present] = (1.0 - beta) / (
            1.0
            - torch.pow(torch.full_like(foreground[present], beta), foreground[present])
        )
        effective[present] /= (
            effective[present].mean().clamp_min(torch.finfo(effective.dtype).eps)
        )
        effective.clamp_(
            min=1.0 / self.class_balance_max_weight,
            max=self.class_balance_max_weight,
        )
        class_weights = torch.cat((torch.ones(1, dtype=torch.float64), effective))
        raw = torch.tensor(
            [
                max((float(class_weights[label]) for label in labels), default=1.0)
                for labels in image_classes
            ],
            dtype=torch.float64,
        )
        raw.div_(raw.mean().clamp_min(torch.finfo(raw.dtype).eps))
        raw.clamp_(max=self.class_balance_max_weight)
        return raw.mul(self.rare_image_sampling_strength).add(
            1.0 - self.rare_image_sampling_strength
        )

    @property
    def num_tasks(self) -> int:
        return NUM_TASKS

    @property
    def train_datasets(self) -> Tuple[Dataset, ...]:
        return tuple(self._train_sets)

    @property
    def validation_datasets(self) -> Tuple[Dataset, ...]:
        return tuple(self._val_sets)

    def _wrap_domains(
        self,
        datasets: Sequence[Dataset],
        split: str,
    ) -> List[Dataset]:
        definitions = (
            TEST_DOMAIN_DEFINITIONS if split == "test" else TRAIN_DOMAIN_DEFINITIONS
        )
        wrapped: List[Dataset] = []
        for domain_id, dataset in enumerate(datasets):
            metadata = DetectionDatasetMetadata(
                domain_id=domain_id,
                domain_name=DOMAIN_NAMES[domain_id],
                split=split,
                definition=definitions[domain_id],
                class_names=CLASS_NAMES,
            )
            wrapped.append(
                DetectionDomainDataset(
                    dataset,
                    metadata,
                )
            )
        return wrapped

    def _ensure_test_sets_loaded(self) -> None:
        if self._test_sets is not None:
            return
        test_sets = self.clad.get_cladd_test(
            self.root_dir,
            transform=self.eval_transform,
            avalanche=False,
        )
        _require_four_domains(test_sets, split="test")
        self._test_sets = self._wrap_domains(test_sets, "test")

    def get_test_image_metadata(self) -> Dict[int, Mapping[str, Any]]:
        """Environmental annotations for exactly the official test images."""
        self._ensure_test_sets_loaded()
        return {
            int(image_id): dataset.official_dataset.img_annotations[image_id]
            for dataset in self._test_sets
            for image_id in dataset.official_dataset.ids
        }

    def _dataset_for(self, domain_id: int, split: str) -> Dataset:
        if not 0 <= domain_id < NUM_TASKS:
            raise ValueError(
                f"Domain {domain_id} is invalid; CLAD-D domains are 0..{NUM_TASKS - 1}."
            )
        if split == "train":
            return self._train_sets[domain_id]
        if split in ("val", "validation"):
            return self._val_sets[domain_id]
        if split == "test":
            self._ensure_test_sets_loaded()
            assert self._test_sets is not None
            return self._test_sets[domain_id]
        raise ValueError("split must be 'train', 'validation', or 'test'")

    def get_domain_dataloader(
        self,
        domain_id: int,
        *,
        split: str,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        dataset = self._dataset_for(domain_id, split)
        is_train = split == "train"
        selected_batch_size = batch_size or (
            self.batch_size if is_train else self.eval_batch_size
        )
        generator = torch.Generator().manual_seed(self.seed + domain_id)
        sampler = None
        if is_train and self.rare_image_sampling and len(dataset) > 0:
            sampler = WeightedRandomSampler(
                self._rare_image_weights(dataset),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        worker_options: Dict[str, Any] = {}
        if self.num_workers > 0:
            worker_options.update(
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )

        return DataLoader(
            dataset,
            batch_size=selected_batch_size,
            shuffle=is_train and len(dataset) > 0 and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.clad.collate_fn_cladd,
            generator=generator,
            **worker_options,
        )

    def get_task_dataloader(
        self,
        task_id: int,
        train: bool = True,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Return the train or same-domain validation loader for one task."""
        split = "train" if train else "validation"
        return self.get_domain_dataloader(
            task_id,
            split=split,
            batch_size=batch_size,
        )

    def get_validation_dataloader(
        self,
        domain_id: int,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        return self.get_domain_dataloader(
            domain_id,
            split="validation",
            batch_size=batch_size,
        )

    def get_all_test_dataloaders(self) -> List[DataLoader]:
        return [self.get_domain_dataloader(i, split="test") for i in range(NUM_TASKS)]

    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        train_dataset = self._dataset_for(task_id, "train")
        val_dataset = self._dataset_for(task_id, "validation")
        metadata = train_dataset.metadata
        official_train_dataset = train_dataset.official_dataset
        class_object_counts: Optional[Dict[int, int]] = None
        image_ids = getattr(official_train_dataset, "ids", None)
        image_annotations = getattr(official_train_dataset, "img_anns", None)
        if image_ids is not None and image_annotations is not None:
            flat_labels = [
                annotation["category_id"]
                for image_id in image_ids
                for annotation in image_annotations[image_id]
            ]
        else:
            load_target = getattr(official_train_dataset, "_load_target", None)
            if callable(load_target):
                targets = [
                    load_target(index) for index in range(len(official_train_dataset))
                ]
            else:
                targets = [
                    official_train_dataset[index][1]
                    for index in range(len(official_train_dataset))
                ]
            flat_labels = [
                int(label)
                for target in targets
                for label in torch.as_tensor(target["labels"]).tolist()
            ]
        labels = torch.as_tensor(flat_labels, dtype=torch.int64)
        if labels.numel() and (
            (labels < 1).any() or (labels > NUM_FOREGROUND_CLASSES).any()
        ):
            raise ValueError("Official CLAD-D dataset contains invalid class IDs")
        counts = torch.bincount(labels, minlength=NUM_DETECTOR_CLASSES)
        class_object_counts = {
            class_id: int(counts[class_id].item()) for class_id in CLASS_NAMES
        }
        return {
            "task_id": task_id,
            "domain_id": task_id,
            "domain_name": metadata.domain_name,
            "domain_definition": dict(metadata.definition),
            "num_train_images": len(train_dataset),
            "num_validation_images": len(val_dataset),
            "num_foreground_classes": NUM_FOREGROUND_CLASSES,
            "num_detector_classes": NUM_DETECTOR_CLASSES,
            "background_class_id": BACKGROUND_CLASS_ID,
            "class_names": dict(CLASS_NAMES),
            # The official ``targets`` property is flattened per object.  It is
            # built from the full annotation index rather than filtered domain
            # IDs, so aggregate counts above deliberately use ``ids`` and
            # ``img_anns`` instead.
            "class_object_counts": class_object_counts,
        }


def create_cladd_benchmark(config: Mapping[str, Any]) -> CLADDBenchmark:
    """Create a detection-only CLAD-D benchmark from the CARL-D config."""
    benchmark_config = config.get("benchmark", {})
    training_config = config.get("training", {})
    data_loading_config = config.get("data_loading", {})

    if "data_root" not in benchmark_config:
        raise ValueError("Missing benchmark.data_root in config.")
    if "clad_repo_root" not in benchmark_config:
        raise ValueError("Missing benchmark.clad_repo_root in config.")

    configured_num_tasks = benchmark_config.get("num_tasks", NUM_TASKS)
    if configured_num_tasks != NUM_TASKS:
        raise ValueError(f"CLAD-D requires benchmark.num_tasks={NUM_TASKS}")
    configured_foreground = benchmark_config.get(
        "num_foreground_classes", NUM_FOREGROUND_CLASSES
    )
    if configured_foreground != NUM_FOREGROUND_CLASSES:
        raise ValueError(
            f"CLAD-D requires benchmark.num_foreground_classes={NUM_FOREGROUND_CLASSES}"
        )
    configured_detector_classes = benchmark_config.get(
        "num_detector_classes", NUM_DETECTOR_CLASSES
    )
    if configured_detector_classes != NUM_DETECTOR_CLASSES:
        raise ValueError(
            f"Faster R-CNN requires {NUM_DETECTOR_CLASSES} outputs: "
            "background plus six CLAD-D foreground classes"
        )
    if (
        int(benchmark_config.get("background_class_id", BACKGROUND_CLASS_ID))
        != BACKGROUND_CLASS_ID
    ):
        raise ValueError("CLAD-D reserves category 0 for detector background")
    configured_ids = tuple(
        int(value)
        for value in benchmark_config.get("foreground_class_ids", tuple(CLASS_NAMES))
    )
    if configured_ids != tuple(CLASS_NAMES):
        raise ValueError("CLAD-D foreground_class_ids must be [1, 2, 3, 4, 5, 6]")
    if "class_names" in benchmark_config:
        configured_names = {
            int(key): str(value)
            for key, value in benchmark_config["class_names"].items()
        }
        if configured_names != dict(CLASS_NAMES):
            raise ValueError("benchmark.class_names do not match official CLAD-D")
    if (
        "domain_names" in benchmark_config
        and tuple(benchmark_config["domain_names"]) != DOMAIN_NAMES
    ):
        raise ValueError("benchmark.domain_names do not match official CLAD-D")

    clad_repo_root = str(benchmark_config["clad_repo_root"])
    os.environ[_CLAD_REPO_ROOT_ENV] = clad_repo_root
    os.environ[_CLAD_SUPPRESS_ENV] = "1"
    clad_module = _import_clad(clad_repo_root)

    return CLADDBenchmark(
        root_dir=str(benchmark_config["data_root"]),
        clad_module=clad_module,
        batch_size=int(training_config.get("batch_size", 2)),
        eval_batch_size=int(training_config.get("eval_batch_size", 2)),
        num_workers=int(data_loading_config.get("num_workers", 4)),
        pin_memory=bool(data_loading_config.get("pin_memory", True)),
        prefetch_factor=int(data_loading_config.get("prefetch_factor", 2)),
        persistent_workers=bool(data_loading_config.get("persistent_workers", True)),
        rare_image_sampling=bool(training_config.get("rare_image_sampling", True)),
        rare_image_sampling_strength=float(
            training_config.get("rare_image_sampling_strength", 0.5)
        ),
        class_balance_beta=float(training_config.get("class_balance_beta", 0.999)),
        class_balance_max_weight=float(
            training_config.get("class_balance_max_weight", 3.0)
        ),
        seed=int(config.get("seed", 0)),
    )


__all__ = [
    "BACKGROUND_CLASS_ID",
    "CLASS_NAMES",
    "CLADDBenchmark",
    "DOMAIN_NAMES",
    "DetectionDatasetMetadata",
    "NUM_DETECTOR_CLASSES",
    "NUM_FOREGROUND_CLASSES",
    "NUM_TASKS",
    "TEST_DOMAIN_DEFINITIONS",
    "TRAIN_DOMAIN_DEFINITIONS",
    "DetectionDomainDataset",
    "create_cladd_benchmark",
]
