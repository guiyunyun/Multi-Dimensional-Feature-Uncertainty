# Active Learning Framework — Multi-Dimensional Feature Uncertainty

An active learning framework built on frozen DINOv3 (ViT-B/16) backbone for image classification. It combines four complementary feature-space uncertainty measures with a closed-loop prediction uncertainty fusion mechanism to select the most informative samples for labelling.

## Project Structure

```
active_learning/
├── main.py                     # Entry point (zero-argument, config-driven)
├── config.py                   # Centralised configuration
├── feature_extractor.py        # Frozen DINOv3 multi-layer feature extractor
├── memory_bank.py              # L2-normalised feature store with KNN queries
├── uncertainty.py              # Four feature-space uncertainties
├── prediction_uncertainty.py   # Prediction uncertainty & attention modulation fusion
├── cascading_selector_v1.py    # Cascading sample selector (used in open-loop only)
├── classifier.py               # Trainable classification head (Linear / MLP)
├── dataset.py                  # Labelled / unlabelled pool management
├── active_learner.py           # Main active learning loop
└── README.md
```

## Two Operating Modes

The framework supports two sample selection strategies. Both share the same training loop; the only difference is how unlabelled samples are selected at each round.

### 1. Random Sampling (Baseline)

```
Unlabelled Pool ──randomly select budget samples──▶ Labelled Pool
                                                        │
                          ┌─────────────────────────────┘
                          ▼
                   Train Classifier
                          │
                          ▼
                   Evaluate on Val Set
                          │
                          ▼
                    Next Round ───▶ ...
```

Enable by setting `random_sampling = True` in `config.py`.

### 2. Closed-Loop Active Learning

The closed-loop pipeline exploits both feature-space signals and classifier feedback:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Per-Round Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Unlabelled Pool                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  Frozen DINOv3 Feature Extractor (ViT-B/16, Layers 3/6/9/11)          │
│       │                                                                 │
│       ├──▶ CLS Features [N, 768]                                       │
│       │         │                                                       │
│       │         ├──▶ KNN Query (K=10) against Memory Bank              │
│       │         │         │                                             │
│       │         │         ├──▶ Exploration  (min dist to labelled)      │
│       │         │         ├──▶ Boundary     (KNN label entropy)        │
│       │         │         └──▶ Density      (KNN similarity std)       │
│       │         │                                                       │
│       │         └──▶ Classifier (768 → 100) ──▶ Softmax                │
│       │                                           │                     │
│       │                                    Prediction Uncertainty       │
│       │                                    (Softmax Entropy u_pred)     │
│       │                                                                 │
│       └──▶ Multi-Layer Features ──▶ Multi-Scale (cross-layer consist.) │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Attention Modulation Fusion (HybridUncertaintyFusion)          │   │
│  │                                                                  │   │
│  │  For each uncertainty i ∈ {exploration, boundary, density, ms}: │   │
│  │    w_i = base_i + boost_i × u_pred                              │   │
│  │  Normalise: w_i = w_i / Σw                                      │   │
│  │  feature_combined = Σ(w_i × u_i)                                │   │
│  │  modulation = mod_min + (mod_max − mod_min) × u_pred            │   │
│  │  final_uncertainty = feature_combined × modulation               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Density Noise Filter (raw density < threshold → keep)                 │
│       │                                                                 │
│       ▼                                                                 │
│  Top-K Selection (highest final_uncertainty, budget samples)           │
│       │                                                                 │
│       ▼                                                                 │
│  Selected samples ──▶ Labelled Pool                                    │
│       │                    │                                            │
│       │                    ├──▶ Train Classifier (next round)          │
│       │                    └──▶ Update Memory Bank                     │
│       ▼                                                                 │
│  Evaluate on Val Set                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  Next Round ───▶ ...                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

Enable by setting `random_sampling = False` and `use_prediction_uncertainty = True` in `config.py`.

## Four Feature-Space Uncertainties

| Uncertainty | Input | Method | Intuition |
|---|---|---|---|
| **Exploration** | CLS features | Min cosine distance to any labelled sample in Memory Bank | Far from known regions → worth exploring |
| **Boundary** | CLS features | Shannon entropy of KNN label distribution | Mixed-label neighbourhood → decision boundary |
| **Density** | CLS features | Std of KNN cosine similarities | Scattered neighbours → isolated / noisy sample |
| **Multi-Scale** | Layer 3/6/9/11 features | Cross-layer uncertainty consistency | Disagreement across abstraction levels → semantically complex |

All four are normalised to [0, 1] via min-max scaling within each batch.

## Attention Modulation Fusion

The fusion mechanism uses prediction uncertainty (classifier softmax entropy) to dynamically adjust the weight of each feature uncertainty:

- **High `u_pred`** (classifier confused): boost Exploration and Boundary weights, suppress Density weight
- **Low `u_pred`** (classifier confident): balanced weights across all four

Default weight configuration:

```
exploration: base=0.25, boost=+0.15   → range [0.25, 0.40]
boundary:    base=0.25, boost=+0.15   → range [0.25, 0.40]
multiscale:  base=0.25, boost=−0.10   → range [0.15, 0.25]
density:     base=0.25, boost=−0.20   → range [0.05, 0.25]
```

After weighted combination, the result is further scaled by a modulation factor `[1.0, 1.5]` proportional to `u_pred`.

## Quick Start

### Prerequisites

```bash
conda activate dinov3
pip install torch torchvision tqdm
```

### Dataset

Prepare ImageNet-100 in the following structure and update `Config.data_root` in `config.py`:

```
ImageNet100/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (100 classes)
└── val/
    ├── n01440764/
    └── ...
```

### Running Experiments

```bash
cd active_learning
python main.py
```

All parameters are controlled via `config.py`. Key settings:

```python
# ---- Baseline: random sampling ----
random_sampling = True
use_prediction_uncertainty = False

# ---- Closed-loop active learning ----
random_sampling = False
use_prediction_uncertainty = True
fusion_strategy = 'attention'

# ---- Ablation: remove a specific uncertainty ----
active_feature_uncertainties = ['exploration', 'boundary', 'multiscale']  # e.g. no density
```

Preset functions are available at the bottom of `config.py`:

```python
set_random_sampling()           # Random baseline
set_closed_loop_full()          # Closed-loop with all 4 uncertainties
set_closed_loop_no_density()    # Ablation: remove density
set_closed_loop_no_boundary()   # Ablation: remove boundary
```

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `'base'` | DINOv3 variant (`small`/`base`/`large`) |
| `feature_layers` | `[3, 6, 9, 11]` | Transformer layers for multi-scale features |
| `k_neighbors` | `10` | K for KNN queries |
| `final_data_percentage` | `5.0` | Target labelling budget as % of training set |
| `total_rounds` | `10` | Number of active learning rounds |
| `classifier_epochs` | `20` | Training epochs per round |
| `noise_threshold` | `0.7` | Density threshold for noise filtering |
| `fusion_strategy` | `'attention'` | Fusion method (`attention`/`multiply`/`add`) |
| `seed` | `42` | Random seed for reproducibility |

## Output

Each experiment creates a timestamped directory under `results/`:

```
results/{exp_name}_{YYYYMMDD_HHMM}/
├── history.json      # Per-round metrics (accuracy, loss, class distribution)
├── checkpoint.pth    # Classifier weights
├── config.json       # Snapshot of all config parameters
└── log.txt           # Console output log
```

## Ablation Experiment Design

| # | Experiment | Feature Uncertainties | Prediction Uncertainty | Purpose |
|---|---|---|---|---|
| 0 | Random Sampling | — | — | Baseline |
| 1 | Closed-loop full | E + B + D + M | Entropy | Full system |
| 2 | No boundary | E + D + M | Entropy | Contribution of boundary signal |
| 3 | No density | E + B + M | Entropy | Contribution of density filtering |
| 4 | No exploration | B + D + M | Entropy | Contribution of exploration signal |
| 5 | No multi-scale | E + B + D | Entropy | Contribution of multi-scale consistency |

## Testing Individual Components

```bash
python feature_extractor.py       # Feature extractor smoke test
python memory_bank.py             # Memory Bank operations
python uncertainty.py             # Four uncertainty calculations
python prediction_uncertainty.py  # Prediction uncertainty & fusion
python cascading_selector_v1.py   # Cascading selector logic
python classifier.py              # Classification head
python dataset.py                 # Dataset loading & pool management
python config.py                  # Print current configuration
```

## License

This project builds upon [DINOv3](https://github.com/facebookresearch/dinov3) by Meta AI Research, licensed under the Apache License 2.0. The active learning components are original work by the author.
