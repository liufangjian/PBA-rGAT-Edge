# PBA-rGAT-Edge
Official PyTorch implementation for DAC 2026 paper **PBA-rGAT-Edge: Arc-Level Path-based Timing Prediction with Scalable Residual Edge-Aware Graph Attention**

[![DAC 2026](https://img.shields.io/badge/DAC-2026-blue)](https://dac.com)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1145/3770743.3810986-orange)](https://doi.org/10.1145/3770743.3810986)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Abstract

Path-Based Analysis (PBA) provides accurate timing by evaluating real signal transitions along true paths, but its computational cost makes full-chip deployment impractical on modern large-scale designs. Graph-Based Analysis (GBA), while much faster, often suffers from pessimism due to worst-case propagation. To bridge this accuracy–efficiency gap, we propose PBA-rGAT-Edge, a scalable residual graph attention model that predicts arc-level delay and slew using edge-aware message passing. Unlike prior cell-centric formulations, PBA-rGAT-Edge operates on a pin-level timing graph and explicitly embeds both cell and interconnect arcs as edge features. A lightweight residual attention backbone, coupled with multi-level feature fusion and dual-task prediction, enables efficient training and accurate timing inference. On three benchmark test cases, the model achieved a 29× to 48× inference speedup compared to commercial STA analysis, while maintaining stable acceleration performance across diverse circuit structures, significantly enhancing timing analysis efficiency.

## 📌 Authors
Fangjian Liu, Chenyang Lv, Chunyang Feng
Shenzhen Giga Design Automation Co., Ltd., Shenzhen, China
{fjliu, cylv, cyfeng}@giga-da.com

## Framework Overview

PBA-rGAT-Edge follows a three-stage workflow (Section 4.1, Fig. 5 in the paper):

1. **Feature and graph construction.** From GBA timing-path reports, extract pin-level electrical features and arc-level cell/net attributes, then construct a pin-based timing graph whose directed edges correspond to timing arcs. This representation enables fine-grained arc-level modeling beyond conventional gate-centric GNNs.
2. **Residual edge-aware attention layers.** A stack of ResAtt-Edge layers performs edge-aware message passing, where node features and arc attributes jointly determine attention coefficients. Each layer applies residual updates to stabilize optimization and expand the effective receptive field. All intermediate node embeddings are retained for multi-scale fusion.
3. **Arc-level prediction head.** Multi-level node embeddings are concatenated and fed into two lightweight MLP heads (Section 4.4) to predict arc slew and delay, providing PBA-quality estimates without invoking full PBA.

## 🔧 Environment Requirements
### Hardware
- GPU: NVIDIA RTX 4080 (≥16GB VRAM)
- CPU: 4× Intel Xeon Platinum 8356H
- RAM: 1 TB DDR4 ECC
- OS: CentOS 7.6
### Software
```bash
python >= 3.9
torch >= 1.13
torch_geometric >= 2.3
numpy
pyyaml
matplotlib
tqdm
```
Install dependencies:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy pyyaml matplotlib tqdm
```

## 📂 Data Generation

In the original paper (Section 5.1), datasets are constructed from **worst-case setup timing paths** reported by a commercial STA tool on 8 industrial designs. The training set comprises 5 designs (Design-1 through Design-5), ranging from 163,226 to 19,064,180 pins and 150,874 to 18,377,535 nets. The test set comprises 3 unseen designs (Design-6, Design-7, Design-8), with 82,886 to 684,548 pins. Node features include *slew_type*, *max_rise/fall_transition_time*, and *direction_of_signal*; edge features include *GBA_delay*, *arc_type*, *max_total_resistance*, and *CCS_total_max_rise/fall_cap* (see Table 1 in the paper).

The open-source codebase provides a **physically compliant synthetic timing graph generator** (`src/data.py`) that follows the same EDA static timing physical laws as the paper: dual-pass Elmore delay calculation, GBA pessimism inflation, and STA worst-path timing propagation rules. Feature dimensions and definitions are fully aligned with Table 1 of the paper.

### Important Data Availability Statement
The industrial timing datasets used in the original paper experiments are proprietary design files exported from commercial EDA static timing tools, protected by internal copyright and non-disclosure agreements, which cannot be publicly distributed.

The synthetic data is provided solely to verify the complete executable workflow and validate the proposed network logic. Numerical results trained on synthetic graphs will differ from the paper's reported R²/MAE values on industrial designs.

## 🚀 Quick Start

### Run Training + Test Evaluation

```bash
python run.py
```

The script first trains the model on synthetic timing graphs, then evaluates the best checkpoint on test cases, and prints a final summary table.

### Custom Configuration

Override config via CLI arguments or `configs/default.yaml`:

```bash
python run.py --epochs 30 --batch-size 64 --name exp1
```

Key configurable parameters (in order of priority: CLI > `configs/default.yaml` > code default):

| Parameter | Config key | Default | Description |
|---|---|---|---|
| `--epochs` | `training.epochs` | 30 | Number of training epochs |
| `--batch-size` | `training.batch_size` | 64 | Training batch size |
| `--train_cases` | `data.train_cases` | design1–design5 | Training case list (comma separated) |
| `--project` | `paths.project` | `./runs/` | Save root directory |
| `--name` | — | `exp` | Experiment name (subdirectory) |
| — | `model.gat_layer_num` | 3 | Number of ResAtt-Edge layers |
| — | `training.lr` | 0.001 | Learning rate |
| — | `training.loss_coefficient.delay` | 0.95 | Delay loss weight |
| — | `training.loss_coefficient.slew` | 0.05 | Slew loss weight |

### Training Algorithm
See Algorithm 1 in the paper for full end-to-end training flow with Huber loss.

## ⚡ Inference / Timing Prediction

Inference runs automatically after training: the best model (saved to `runs/pbargatedge_1108/pbargatedge_best.pt`) is evaluated on test cases and results are printed in the final summary table. Per-case metrics and predictions are also saved to `output/pbargatedge/test/`.

## 📊 Reproduce Paper Experiments

The paper's main results (Table 2, Table 3) were obtained on 8 proprietary industrial designs (Section 5.1) processed through a commercial STA tool. The open-source code provides a synthetic data generator (`src/data.py`) with matching feature definitions and graph structure for workflow validation.

### Baseline Comparison (Section 5.1, Table 2)
PBA-rGAT-Edge is compared against five baselines — MLP, GraphSAGE, GAT, EGNN, and DeepEdgeGAT — each independently re-implemented as described in Section 5.1. See paper Table 2 for full comparison across all metrics.

### Ablation Studies (Section 5.2)
The paper ablates three design choices:
1. **Edge feature aggregation** (Fig. 7b): removing edge features during aggregation leads to non-convergence
2. **Residual connections**: removing them causes significant performance degradation across all metrics
3. **Layer depth** (Fig. 6c): 5-layer configuration achieves the best accuracy-efficiency trade-off

## 📈 Main Experimental Results

**Table 2 — Unified comparison with baselines on industrial testcases (Section 5.1, paper Table 2):**

| Metric | MLP | GraphSAGE | GAT | EGNN | DeepEdgeGAT | **PBA-rGAT-Edge** |
|---|---|---|---|---|---|---|
| Avg Slew R² | 0.946 | 0.941 | 0.972 | 0.969 | 0.845 | **0.974** |
| Avg Delay R² | 0.987 | 0.996 | 0.997 | 0.997 | 0.996 | **0.998** |
| Path R² | 0.986 | 0.992 | 0.995 | 0.997 | 0.996 | **0.999** |
| Path MAE (ps) | 15.464 | 17.027 | 9.720 | 6.855 | 8.973 | **3.705** |

**Table 3 — Training epochs and runtime (paper Table 3):**

| Metric | MLP | GraphSAGE | GAT | EGNN | DeepEdgeGAT | **PBA-rGAT-Edge** |
|---|---|---|---|---|---|---|
| Epochs to converge | 75 | 235 | 802 | 197 | 183 | **9** |
| Total train time (s) | 558 | 2,333 | 11,124 | 3,292 | 45,116 | 1,669 |

Key findings:
1. **Accuracy** — Average arc delay R²: **0.998**; Path delay R²: **0.999**, average Path MAE: **3.705 ps** (max ≤4.2 ps across test cases)
2. **Convergence** — Only **9 training epochs** to converge, far faster than GAT (802 epochs) / DeepEdgeGAT (183 epochs)
3. **Speed** — **29× to 48× inference speedup** compared to commercial full PBA static timing analysis (Design-6: 29×, Design-7: 45×, Design-8: 48×)
4. **Scalability** — Stable performance on designs with up to ~19M pins

## 📝 Citation
If you find this work useful in your research or industrial flow, please cite our DAC 2026 paper:
```bibtex
@inproceedings{liu2026pbargatedge,
  title={PBA-rGAT-Edge: Arc-Level Path-based Timing Prediction with Scalable Residual Edge-Aware Graph Attention},
  author={Fangjian Liu, Chenyang Lv, Chunyang Feng},
  booktitle={63rd ACM/IEEE Design Automation Conference (DAC '26)},
  year={2026},
  doi={10.1145/3770743.3810986}
}
```

## ⚠️ Errata

**Equation (4) dimension in the original paper**: Eq. (4) in Section 4.3 specifies the encoder weight as $W_{\text{enc}} \in \mathbb{R}^{D_n \times (D_n + D_e)}$, not $W_{\text{enc}} \in \mathbb{R}^{D_n \times (2D_n + D_e)}$ as printed. The aggregated vector $g_i^d$ has dimension $D_n + D_e$ (concatenation of $x_j^{d-1}$ and $e_{ij}$), so the correct mapping is $D_n + D_e \to D_n$.

In this codebase, the `ResidualEncoder` at [`src/model.py`](src/model.py) implements the correct dimension:

```python
nn.Linear(node_feat_dim + edge_feat_dim, node_feat_dim)
```

The full encoder follows Eq. (4) faithfully: `Linear(D_n + D_e, D_n) → LayerNorm → LeakyReLU`.

**Section 5.1 (Speedup range)**: The inference speedup over commercial STA PBA is **29× to 48×** per-testcase (Design-6: 29×, Design-7: 45×, Design-8: 48×), corrected from any condensed range notation in the original text.

**Abstract (Open-source URL)**: The open-source URL listed in the paper's abstract is `github.com/PBA-rGAT-Edge`; the correct repository is `https://github.com/liufangjian/PBA-rGAT-Edge`.

## ❓ FAQ

1. **Where can I get the industrial timing dataset?**
   Our dataset is confidential industrial STA output. The open-source code provides a synthetic timing graph generator (`src/data.py`) for workflow validation.

2. **How to tune hyperparameters for smaller designs?**
   Reduce `gat_layer_num` to 3–4 and lower batch size in `configs/default.yaml`.

3. **Why do experimental metrics differ from the DAC 2026 paper values?**
   The paper's results were obtained on proprietary industrial designs. The synthetic data follows the same physical timing rules and feature definitions but uses artificially generated electrical parameters, so R²/MAE values will differ.

## 📄 License
This project is released under the MIT License. See [LICENSE](LICENSE) file for full terms.