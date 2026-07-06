# QIAFP: Quantum Imbalance-Aware Feature Projection

**Paper:** "Quantum-Assisted Anomaly Detection for Imbalanced Environmental IoT Data: A Rigorous Empirical Evaluation"  
**Authors:** Sai Suganda Sutar, Dr. Geetika Vashishta  
**Affiliation:** Department of Computer Science, College of Vocational Studies, University of Delhi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saisuganda/Extended-qiafp-iot-anomaly/blob/main/Extended_QIAFP_Experiments%20.ipynb.ipynb)
## Overview

This repository contains the complete experiment code for the QIAFP paper. The notebook evaluates hybrid classical–quantum feature fusion for anomaly detection across three IoT datasets:

* **Intel Berkeley Research Lab** (environmental sensor, synthetic anomaly injection)
* **UCI Air Quality** (chemical sensor array, synthetic anomaly injection)
* **SKAB** (Skoltech industrial IoT water-cooling, expert-labeled real anomalies)

## Key Findings

* Hybrid QIAFP features **significantly degrade** Isolation Forest on all 3 datasets (p ≤ 0.004), confirmed across 10 QIAFP parameter seeds
* Autoencoder shows the **only significant positive effect** under hybrid fusion (Precision +0.054, p = 0.047 on Air Quality)
* Quantum-only features degrade all detectors across all datasets
* Fisher separability is preserved within 4% by hybrid fusion, but preservation is **necessary not sufficient** for detection improvement

## How to Run

### Option 1: Google Colab (recommended)
Click the "Open in Colab" badge above. Run all cells top to bottom.

**Requirements handled automatically by Cell 1:**
pennylane pennylane-lightning pennylane-lightning-gpu
scikit-learn pandas numpy matplotlib seaborn
ucimlrepo torch scipy kaggle

**For NASA SMAP dataset** (optional third dataset): upload your `kaggle.json` API token before running Cell 5.

### Option 2: Local / GPU server

```bash
git clone https://github.com/saisuganda/qiafp-iot-anomaly.git
cd qiafp-iot-anomaly
pip install -r requirements.txt
jupyter notebook QIAFP_Experiments_v2.ipynb
```

## Repository Structure
qiafp-iot-anomaly/

├── QIAFP_Experiments_v2.ipynb   # Main experiment notebook (21 cells)

├── requirements.txt              # Python dependencies

├── LICENSE                       # MIT License

└── README.md                     # This file

## Experiment Outputs

Running the notebook end-to-end produces `qiafp_results/`:

| File | Contents |
|------|----------|
| `all_results.json` | Raw per-seed metrics for all detectors × datasets × feature conditions |
| `significance.csv` | Wilcoxon signed-rank test results |
| `fisher_separability.csv` | Fisher S values for all feature conditions |
| `imbalance_ratio.csv` | Performance at 1:10, 1:50, 1:100 imbalance |
| `complexity_benchmark.csv` | QIAFP runtime vs N |
| `tsne_intel_berkeley.pdf` | t-SNE figure (Intel) |
| `tsne_air_quality_uci.pdf` | t-SNE figure (Air Quality) |
| `tsne_skab.pdf` | t-SNE figure (SKAB) |

## Hardware

Experiments were run on Google Colab with NVIDIA GPU (T4/A100).  
PennyLane backend: `lightning.gpu` → `lightning.qubit` → `default.qubit` (auto-fallback).

## License

MIT License — see `LICENSE` for details.
