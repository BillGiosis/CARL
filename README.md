# CARL
Implementation of Continual Adaptive Reinforcement Learning (CARL) for continual object detection on
[CLAD-D](https://github.com/VerwimpEli/CLAD).

CARL uses a COCO-pretrained Faster R-CNN–ResNet50-FPN V2 detector with a shared
prediction head across four sequential domains. No task identifier is supplied
at inference. A DQN controller selects adaptation actions during the continual
stream, subject to parameter limits and action cooldowns.

## Method overview

- **Structural adaptation:** zero-initialized parallel residual adapters in the
  backbone, feature pyramid network (FPN), and region-of-interest (RoI) head.
- **Replay:** a 250-image buffer with domain/class-aware admission and sampling.
  Training uses retained images from completed domains only; Task 1 has no replay.
- **Replay distillation:** cached proposals and teacher logits provide
  temperature-scaled KL supervision on matched replay RoIs. No persistent teacher
  network is required during replay training.
- **Imbalance handling:** class-aware current-image sampling, Class-Balanced
  positive-RoI classification loss, and balanced replay. Background RoIs retain
  unit loss weight.
- **Learned control:** adapter expansion, temporary backbone freezing, replay-loss
  weighting, retained-buffer refinement, and a no-operation action.

Stagewise learning-rate multipliers protect pretrained backbone features after
Task 1. Temporary controller-triggered freezes leave backbone adapters trainable.
The parameter-growth ceiling is a method constraint, separate from the dataset
protocol.

## Installation

Use an isolated Python environment with a compatible PyTorch/Torchvision
installation. CUDA is recommended for full training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a matching PyTorch/Torchvision build for the target CUDA environment.
python -m pip install -r requirements.txt
```

The dependency file specifies version bounds, not a locked environment. Record
the resolved package versions, CUDA version, and hardware for each experiment.
Initial model construction may download pretrained COCO weights.

## Dataset preparation

Obtain the dataset according to the instructions and terms of the
[official CLAD repository](https://github.com/VerwimpEli/CLAD). Dataset images and
annotations are not distributed with this implementation. The official loader
defines the train/validation/test splits and transforms.

The expected directory structure is:

```text
<dataset-root>/
└── SSLAD-2D/
    └── labeled/
        ├── train/
        ├── val/
        ├── test/
        └── annotations/
            ├── instance_train.json
            ├── instance_val.json
            └── instance_test.json
```

Set the following entries in [config/carl_d.yaml](config/carl_d.yaml), replacing
the example paths with local absolute paths:

```yaml
benchmark:
  data_root: /path/to/dataset-root
  clad_repo_root: /path/to/CLAD
```

The six foreground classes are Pedestrian, Cyclist, Car, Truck, Tram (Bus), and
Tricycle, with labels 1–6. Label 0 is detector background.

| Task | Domain label | Training images | Validation images | Test images |
| --- | --- | ---: | ---: | ---: |
| 1 | Clear daytime city street | 4,470 | 497 | 2,433 |
| 2 | Daytime highway | 1,329 | 148 | 3,126 |
| 3 | Night | 1,479 | 165 | 2,968 |
| 4 | Rainy daytime | 524 | 59 | 1,442 |
| Total | | 7,802 | 869 | 9,969 |

Use the official loader definitions rather than reconstructing splits from these
short labels.

## Training

Run commands from the repository root:

```bash
python experiments/train_carld.py --config config/carl_d.yaml --device cuda
```

The default recipe uses 50 epochs per domain, SGD with learning rate 0.005,
momentum 0.9, and weight decay 0.0005. MultiStepLR decays the learning rate at
epochs 34 and 44 within each domain, following one warmup epoch. Images are
resized to a shorter side of 800 pixels with a maximum longer side of 1,333
pixels. Current-image, evaluation, and replay batch sizes default to 16.

DQN feedback is collected every two epochs. It uses annotation-support weighting
`n / (n + 20)` and temporal smoothing for controller state and reward. This
feedback is distinct from the unweighted benchmark AP reported for comparison.
Validation is used by the controller; test data are not used for action selection.

The YAML exposes experiment and runtime controls. Fixed protocol values are
resolved internally and recorded in the effective checkpoint configuration:
ResNet50-FPN V2, parallel adapters, seven detector outputs, SGD/MultiStepLR,
FP16 replay-logit targets, a 250-image replay budget, and COCO 101-point AP50.

## Ablations

Create a separate configuration and output directory for each variant. Apply
the changes below to the full recipe, leaving other settings unchanged.

| Variant | Configuration changes |
| --- | --- |
| CARL-NA: no adapter expansion | `model.adapters.enabled: false` |
| CARL-NB: no imbalance-handling package | Set `model.class_balanced_roi_loss`, `training.rare_image_sampling`, `replay_buffer.class_balanced_admission`, `replay_buffer.balanced_sampling`, and `replay_buffer.batch_aware_sampling` to `false` |
| CARL-ND: no replay distillation | `distillation.enabled: false` |
| CARL-NR: no image replay | `replay_buffer.enabled: false`; replay-dependent distillation is disabled automatically |
| CARL-FS: fixed schedule | `controller.algorithm: fixed_schedule` |

CARL-NA retains the DQN and its remaining actions. CARL-ND retains supervised
replay. CARL-NR retains structural adaptation and current-image imbalance
handling.

## Evaluation protocol

The primary metric is **COCO 101-point bounding-box AP at IoU 0.50**, averaged
equally across the four official test domains. For domain (d), let
(C_d) contain its supported foreground classes:

```text
domain_mAP50[d] = mean(AP50[d, c] for c in C_d)
overall_mAP50   = mean(domain_mAP50[d] for the four official domains)
per_class_AP50  = equal-domain mean of the class's AP50
```

Background is excluded. Classes without ground-truth support receive undefined
AP and are omitted from the corresponding mean, rather than assigned zero.
All official test images are evaluated, including images without predictions.
The evaluator uses the all-object-size range and a maximum-detection setting of
100. Default detector post-processing uses score threshold 0.05 and NMS IoU 0.50.

Reports also include the following metadata-defined subsets:

| Subset | Annotation condition | Test images |
| --- | --- | ---: |
| Day | `period == Daytime` | 7,001 |
| Night | `period == Night` | 2,968 |
| Highway | `location == Highway` | 4,381 |
| City | `location == Citystreet` | 4,560 |
| Country | `location == Countryroad` | 1,028 |

AP is recomputed from predictions within each subset, without another detector
inference pass. Time and location subsets overlap: **Overall is not the mean of
these five columns**. Per-class summary scores remain equally averaged across
the four official domains.

Final test evaluation is integrated into training. To evaluate an existing
compatible checkpoint without retraining:

```bash
python experiments/evaluate_carl_d.py \
  --config config/carl_d.yaml \
  --checkpoint checkpoints/carl/final_DDMMYYYY_HHMMSS.pt \
  --device cuda
```

Use the same evaluator, image subsets, class mapping, and post-processing
conventions for method comparisons. Select configurations using validation data,
not final-test performance. For repeated runs, use matched seed identifiers and
report mean and standard deviation.

## Outputs and checkpointing

The default output directory is `checkpoints/carl`, controlled by
`logging.save_dir`. Training produces:

- `final_DDMMYYYY_HHMMSS.pt`: final model and training state.
- `results_DDMMYYYY_HHMMSS.txt`: training summary and final test results.

Timestamps use 24-hour time. Reports contain training time, task-end validation
history, forgetting, backward transfer, plasticity, class/domain/environment
AP50, parameter and replay statistics, and controller action summaries.
Per-epoch loss records are not included in the text summary. Standalone
evaluation writes a separate `test_results_*.txt` file.

Only the final checkpoint is saved by default. For task-boundary recovery,
enable `checkpointing.save_every_task`. Resume with the same experiment settings:

```bash
python experiments/train_carld.py \
  --config config/carl_d.yaml \
  --resume /path/to/task_checkpoint.pt \
  --device cuda
```

Resume is configuration-strict and supported at task boundaries, not mid-epoch.
Checkpoints store detector/adapter structure, optimizer, scheduler, scaler,
replay, controller, metric history, and RNG state.

## Runtime and reproducibility

- CUDA training uses automatic mixed precision and gradient scaling, with no
  gradient accumulation.
- The ResNet body and FPN use compilation for training. Expandable adapters and
  detection-specific operations remain eager.
- Progress bars are enabled. Input prefetching and replay caches reduce repeated
  data preparation.
- Training is single-device; distributed training is not implemented.

## Repository structure

```text
agents/       DQN and fixed-schedule controller bookkeeping
benchmarks/   official CLAD-D loader integration
buffers/      full-image replay and refinement
config/       experiment configuration
experiments/  training, checkpoint evaluation, and report export
models/       detector, Class-Balanced RoI loss, and parallel adapters
trainers/     continual training, action execution, and configuration validation
utils/        evaluation metrics, reports, and checkpoint serialization
```

## External resources

This implementation uses the [official CLAD repository](https://github.com/VerwimpEli/CLAD)
for its dataset loader and Torchvision's pretrained detector.
