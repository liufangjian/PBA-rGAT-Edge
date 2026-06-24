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
- GPU: NVIDIA RTX 4080 / A100 / H100 (≥16GB VRAM)
- CPU: Multi-core Xeon, ≥256GB RAM for large industrial designs
### Software
```bash
python >= 3.9
torch >= 1.13
torch_geometric >= 2.3
numpy
pandas
scipy
matplotlib
h5py
tqdm
```
Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy pandas scipy matplotlib h5py tqdm
```

## 📂 Dataset Format
We use industrial STA timing datasets exported from commercial EDA tools.
Each design outputs an HDF5 file containing three components:
1. `node_feat`: Pin-level electrical attributes (Table 1 in paper)
   - slew_type, max_rise_trans, max_fall_trans, pin_direction
2. `edge_feat`: Timing arc attributes (cell arc / net arc)
   - GBA_delay, arc_type, total_R, rise_cap, fall_cap
3. `edge_index`: Directed graph connectivity (source pin → dest pin)
4. `label_slew` / `label_delay`: Ground-truth PBA arc slew & delay

### Data Preprocessing Script
`scripts/build_timing_graph.py` converts raw STA report to standardized HDF5 graph file.
```bash
python scripts/build_timing_graph.py --sta_report ./data/design7_gba.rpt --output ./dataset/design7.h5
```

## 🚀 Training
### Step 1: Basic Training Command
```bash
python train.py \
  --train_dataset_root ./dataset/train \
  --test_dataset_root ./dataset/test \
  --layer_num 5 \
  --beta 0.5 \
  --gpu 0 \
  --save_ckpt ./ckpts/pba_rgat_edge_best.pth \
  --log_dir ./logs/train_run
```
Key arguments:
- `--layer_num`: Number of ResAtt-Edge layers (optimal=5 per ablation)
- `--beta`: Weight balance between slew loss & delay loss
- `--early_stop`: Enable early stopping to avoid overfitting
- `--lr_scheduler ReduceLROnPlateau`

### Training Algorithm
See Algorithm 1 in the paper for full end-to-end training flow with Huber loss.

## ⚡ Inference / Timing Prediction
Load trained checkpoint and predict PBA-quality arc delays for unseen industrial designs:
```bash
python infer.py \
  --ckpt ./ckpts/pba_rgat_edge_best.pth \
  --data ./dataset/test/design8.h5 \
  --output_rpt ./result/design8_pred_pba.rpt
```
The output report reconstructs full path delays by summing predicted per-arc values, compatible with standard STA timing closure flow.

## 📊 Reproduce Paper Experiments
### Baseline Comparison
We re-implement all baselines for fair comparison:
- MLP, GraphSAGE, GAT, EGNN, DeepEdgeGAT
Run unified benchmark:
```bash
bash scripts/run_all_baselines.sh
```
### Ablation Studies
1. Edge feature ablation (remove edge aggregation)
2. Residual connection ablation
3. Layer depth sweep (1/5/10/30 layers)
```bash
python ablation_agg.py
python ablation_residual.py
python ablation_layer_depth.py
```
### Speedup Evaluation
Measure inference runtime vs commercial full PBA STA:
```bash
python eval_speedup.py --tool_pba_log ./runtime/sta_pba.log --model_log ./runtime/model_infer.log
```

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

## ❓ FAQ
1. Where can I get the industrial timing dataset?
   Our dataset is confidential industrial STA output. You can generate your own dataset from commercial EDA GBA/PBA reports with our preprocessing script.
2. Can this model be integrated into open-source STA tools (OpenTimer, OpenSTA)?
   Yes, the inference script outputs standard timing report format that can be parsed by open-source STA platforms.
3. How to tune hyperparameters for smaller designs?
   Reduce `layer_num` to 3–4 and lower batch size for small-scale circuits.

## 📄 License
This project is released under the MIT License. See [LICENSE](LICENSE) file for full terms.
```
