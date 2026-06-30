# PBA-rGAT-Edge
Official PyTorch implementation for DAC 2026 paper **PBA-rGAT-Edge: Arc-Level Path-based Timing Prediction with Scalable Residual Edge-Aware Graph Attention**

[![DAC 2026](https://img.shields.io/badge/DAC-2026-blue)](https://dac.com)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1145/3770743.3810986-orange)](https://doi.org/10.1145/3770743.3810986)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Abstract
Path-Based Analysis (PBA) delivers high-accuracy static timing results but suffers extreme runtime overhead on large industrial SoCs, while Graph-Based Analysis (GBA) runs fast yet introduces severe timing pessimism.
We propose **PBA-rGAT-Edge**, a residual edge-aware graph attention network that directly models **cell & net timing arcs** as graph edges to predict PBA-quality slew/delay without full PBA execution:
1. Arc-native graph formulation: explicitly encode pin node features + timing arc edge features;
2. Residual edge-aware attention aggregation fuses both node & edge information;
3. Multi-layer embedding fusion + dual-task head predicts arc slew and delay simultaneously;
4. Up to **29×~48× speedup vs commercial STA PBA**, path delay prediction R²=0.999, MAE ≤4.2ps on industrial testcases.

## 📌 Authors
Fangjian Liu, Chenyang Lv, Chunyang Feng
Shenzhen Giga Design Automation Co., Ltd.

## 🧩 Framework Overview
![Pipeline](https://p-flow-sign.bytedance.net/tos-cn-i-ik7evvg4ik/8ca960e7e16c4aaf9b090cab351a66e9.pdf#page=3)
1. Input: Netlist + GBA timing report, extract pin/node features & arc/edge features
2. Graph Construction: Build pin-level directed timing graph (edge = timing arc)
3. ResAtt-Edge Layers: Edge-aware attention + residual connection stack
4. Multi-level feature concatenation
5. Dual MLP heads predict per-arc slew & delay, sum arcs to full path delay

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

The codebase integrates a **physically compliant synthetic timing graph generator** (`src/data.py`) that follows EDA static timing physical laws: dual-pass Elmore delay calculation, GBA pessimism inflation, STA worst-path timing propagation rules. Feature dimensions and definitions fully aligned with Table 1 of the paper.

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

Inference runs automatically after training: the best model (saved to `runs/rgat_1108/rgat_best.pt`) is evaluated on test cases and results are printed in the final summary table. Per-case metrics and predictions are also saved to `output/rgat/test/`.

## 📊 Reproduce Paper Experiments

The paper's main results (Table 2, Table 3) were obtained on proprietary industrial datasets. The open-source code provides a synthetic data generator (`src/data.py`) with matching feature definitions for workflow validation. See paper Sections 5.1–5.3 for detailed experimental setup.

### Baseline Comparison
The paper compares against MLP, GraphSAGE, GAT, EGNN, and DeepEdgeGAT. See paper Table 2 for full comparison.

### Ablation Studies
The paper ablates (Section 5.2):
1. **Edge feature aggregation**: removing edge features during aggregation leads to non-convergence (Fig. 7b)
2. **Residual connections**: removing them causes significant performance degradation
3. **Layer depth**: 5-layer configuration achieves best trade-off (Fig. 6c)

## 📈 Main Experimental Results
1. **Accuracy**
   - Average arc delay R²: 0.998; Path delay R²: 0.999, MAE ≤4.2ps
2. **Convergence**
   Only 9 training epochs to converge, far faster than GAT / DeepEdgeGAT
3. **Speed**
   29× ~ 48× inference speedup compared to commercial full PBA static timing analysis
4. Scalability: Stable performance on designs with up to ~19M pins

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

## ❓ FAQ

1. **Where can I get the industrial timing dataset?**
   Our dataset is confidential industrial STA output. The open-source code provides a synthetic timing graph generator (`src/data.py`) for workflow validation.

2. **How to tune hyperparameters for smaller designs?**
   Reduce `gat_layer_num` to 3–4 and lower batch size in `configs/default.yaml`.

3. **Why do experimental metrics differ from the DAC 2026 paper values?**
   The paper's results were obtained on proprietary industrial designs. The synthetic data follows the same physical timing rules and feature definitions but uses artificially generated electrical parameters, so R²/MAE values will differ.

## 📄 License
This project is released under the MIT License. See [LICENSE](LICENSE) file for full terms.