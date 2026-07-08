# Intention-Driven Match Phase Identification

Code and example data for the manuscript:

**Intention Driven Identification of In-Possession Match Phases in Association Football through Temporal Graph Learning**

This repository implements a hierarchical framework for identifying in-possession match phases in association football from spatiotemporal tracking-derived features. The method first predicts the high-level tactical intention of each stable possession sequence, then classifies the corresponding fine-grained phase within that intention.

## Overview

The phase model contains three tactical intentions and six in-possession phases:

| Intention | Phase labels |
| --- | --- |
| Invade opponent space | Build Up, Progression, Counter attack |
| Keep possession | Maintenance |
| Scoring | Sustained Threat, Finishing |

The main model is a Temporal Graph Attention Network (T-GAN). Frame-level player interaction graphs are encoded with graph neural network layers, aggregated with player attention pooling, and passed through a Transformer encoder to model the temporal context of possession sequences. The package also contains baseline models using a pure GNN, a pure Transformer, Random Forest classifiers, and a rule-based baseline.

Post-processing utilities are included to convert frame-wise predictions into more coherent tactical sequences using temporal filtering and rule-based correction.

## Repository Structure

```text
phase_model_pkg/
|-- Demo_Datasets/                 # Processed example data for three matches
|-- T_GAN/                         # Temporal Graph Attention Network model, training, prediction
|-- Pure_GNN/                      # GNN-only baseline
|-- Pure_Transformer/              # Transformer-only baseline
|-- Random_Forest/                 # Classical ML baseline
|-- Rule_based/                    # Rule-based baseline
|-- tools/                         # Datasets, losses, training, filtering, fixing, evaluation helpers
|-- GNNDataConverter.py            # Convert CSV graph features to PyTorch Geometric HeteroData batches
`-- Sequence_level_evaluate.py     # Sequence-level IoT-D/tIoU evaluation utilities
```

## Data

The full study used seven German Bundesliga matches recorded at 25 Hz with TRACAB optical tracking data. The authors are not allowed to redistribute the full original data.

This code package contains processed example data in `Demo_Datasets/` for demonstration and smoke testing. Each demo match folder contains:

| File pattern | Description |
| --- | --- |
| `Feature_demo_*.csv` | Frame-level contextual and engineered features with labels |
| `Global_demo_*.csv` | Global frame-level graph/context features |
| `Node_demo_*.csv` | Player/node features for graph construction |
| `Edge_demo_*.csv` | Pairwise player edge features |
| `Action_demo_*.csv` | On-ball action features aligned to frames |
| `Label_demo_*.csv` | Frame-level intention and phase labels |
| `Hetero_demo_*.pt` | Preconverted PyTorch Geometric batched graph data |

The demo data are intended to illustrate the expected input format and to test the code pipeline. They should not be interpreted as the complete dataset used for all reported manuscript results.

Before public release, confirm that the example files are consistent with the data-sharing agreement for the underlying tracking/event provider.

## Installation

The code was tested with Python 3.9. The main dependencies are:

```text
python==3.9
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
torch==2.1.2
torch-geometric==2.6.1
tqdm==4.64.0
matplotlib==3.9.4
seaborn==0.13.2
```

Create an environment and install dependencies, for example:

```bash
conda create -n match-phases python=3.9
conda activate match-phases
pip install numpy pandas scikit-learn tqdm matplotlib seaborn
pip install torch==2.1.2
pip install torch-geometric==2.6.1
```

Depending on your CUDA/PyTorch setup, `torch` and `torch-geometric` may require platform-specific installation commands.

When running scripts from the model subfolders, expose the parent project directory through `PYTHONPATH`. In the examples below, commands are run from `phase_model_pkg/` or one of its subfolders.

## Quick Start

### 1. Convert CSV graph features to PyTorch Geometric data

The demo package already includes `Hetero_demo_*.pt` files. To regenerate them from the CSV files:

```bash
cd phase_model_pkg
PYTHONPATH=.. python GNNDataConverter.py
```

This reads the files in `Demo_Datasets/` and overwrites the corresponding `Hetero_demo_*.pt` files.

### 2. Train the T-GAN model on demo data

```bash
cd phase_model_pkg/T_GAN
PYTHONPATH=../.. python training_T-GAN.py
```

This script performs leave-one-match-out training over the available demo matches. It trains three hierarchical classifiers:

- intention classifier: `Invade opponent space`, `Keep possession`, `Scoring`
- invade-phase classifier: `Build Up`, `Progression`, `Counter attack`
- scoring-phase classifier: `Sustained Threat`, `Finishing`

The script writes model checkpoints and prediction CSV files into the `T_GAN/` folder, for example:

```text
aim_TGAN_demo_<match>.pt
Invade_TGAN_demo_<match>.pt
Scoring_TGAN_demo_<match>.pt
Matchphases_<match>_Phase_pred_T-GAN_demo.csv
Test_f1_per_match_T-GNN.csv
```

### 3. Run prediction with trained T-GAN checkpoints

`predict_T-GAN.py` expects trained checkpoint files for the selected match to be present in `T_GAN/`.

```bash
cd phase_model_pkg/T_GAN
PYTHONPATH=../.. python predict_T-GAN.py
```

By default, the script predicts the demo match `BVB_FCA`. To use another match, edit the `match_name` and `match_team` variables near the top of `predict_T-GAN.py`.

### 4. Train baseline models

Pure GNN baseline:

```bash
cd phase_model_pkg/Pure_GNN
PYTHONPATH=../.. python training_Pure_GNN.py
```

Pure Transformer baseline:

```bash
cd phase_model_pkg/Pure_Transformer
PYTHONPATH=../.. python training_Pure_Transformer.py
```

Random Forest baseline:

```bash
cd phase_model_pkg/Random_Forest
PYTHONPATH=../.. python Phase_identification.py
```

Rule-based baseline:

```bash
cd phase_model_pkg/Rule_based
PYTHONPATH=../.. python Phase_identification_rule_based.py
```

## Evaluation

Frame-level F1 scores are printed during training and saved as `Test_f1_per_match_*.csv` files.

`Sequence_level_evaluate.py` computes sequence-level matrices and summary tables using IoT-D/tIoU-style agreement between ground-truth and predicted tactical segments. It expects prediction CSV files organised in model-specific subfolders under a prediction root. It supports both the full leave-one-match-out prediction naming convention and the current demo output naming convention.

Example:

```bash
cd phase_model_pkg
PYTHONPATH=.. python Sequence_level_evaluate.py \
  --pred-root . \
  --output-root evaluation_results \
  --variant demo
```

Note: the manuscript-level evaluation used seven matches. The bundled `Demo_Datasets/` currently contains three processed example matches, so the sequence-level script may need the `MATCHES` list in `Sequence_level_evaluate.py` to be restricted to the available demo matches, or the remaining full prediction files must be supplied.

## Outputs

The main prediction files contain frame-level and post-processed labels, including:

| Column | Meaning |
| --- | --- |
| `Aim_pred` | Raw predicted tactical intention |
| `Aim_filtered` | Temporally filtered intention prediction |
| `Aim_filtered_indices` | Segment index for filtered intention sequences |
| `Phase_pred_filtered_aim` | Raw phase prediction within the filtered intention |
| `Phase_pred_filtered_aim_filtered` | Temporally filtered phase prediction |
| `Phase_pred_filtered_aim_filtered_fixed` | Phase prediction after filtering and rule-based correction |
| `Phase_pred_filtered_aim_filtered_fixed_indices` | Segment index for final phase sequences |
| `event_importance_intention` | Optional attention/importance score for intention prediction |
| `event_importance_phase` | Optional attention/importance score for phase prediction |

## Reproducibility Notes

- The demo scripts use leave-one-match-out training over the available example matches.
- Training neural models is stochastic; exact scores can vary unless all random seeds and deterministic backend settings are controlled.
- The full manuscript results require the complete seven-match dataset and the same preprocessing pipeline.
- Some filenames contain non-ASCII team names. Avoid renaming demo folders/files unless the corresponding script variables are updated consistently.
- Large intermediate files such as `*.pt` checkpoints and prediction CSVs can be regenerated from the scripts.

## Citation

If you use this code, please cite the associated manuscript:

```text
Li, Y., & Link, D. Intention Driven Identification of In-Possession Match Phases in Association Football through Temporal Graph Learning.
```

Update this citation with the final journal information after publication.

## License and Data Availability

The source code is released under the MIT License. See `LICENSE` for details.

The full original tracking/event data cannot be redistributed by the authors. Only processed example data are included to demonstrate the expected input format and code usage. Users who wish to reproduce the full study need access to the underlying match data and must follow the corresponding data provider's licensing terms.
