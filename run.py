"""
PBA-rGAT-Edge entry point: train → evaluate best model → print summary.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from src.train import run_pipeline, node_feat_dim, edge_feat_dim
from src.test import evaluate_test


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "configs" / "default.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
except Exception:
    CFG = {}


def _cfg(key, default=None):
    """Safely nested access into CFG dict."""
    keys = key.split(".")
    val = CFG
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, {})
        else:
            return default
    return val if val != {} else default


# ---------------------------------------------------------------------------
# Utility: increment path
# ---------------------------------------------------------------------------
def increment_path(path, exist_ok=False, sep=""):
    """Increment file or directory path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc."""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = Path(f"{path}{sep}{n}{suffix}")
            if not p.exists():
                path = p
                break
    return path


def main(opt):
    """Orchestrate training → test evaluation → summary."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    opt.project = str(opt.project)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))

    # ---- Train ----------------------------------------------------------------
    print("=" * 100)
    print("  Phase 1: Training")
    print("=" * 100)
    model, best_val_slew_r2, best_val_r2, best_epoch, model_save_path, epochs = \
        run_pipeline(opt, device)

    # ---- Evaluate best model on test set --------------------------------------
    print("\n" + "=" * 100)
    print("  Phase 2: Evaluating best model on test set")
    print("=" * 100)
    test_case_names = _cfg("data.test_cases", ["design6", "design7", "design8"])

    best_model_path = os.path.join(model_save_path, "pbargatedge_best.pt")
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, map_location=device, weights_only=False)
    else:
        print("  !! Best model not found, using final model.")
        best_model = model

    # Generate test data (use train.py's already-loaded data module)
    data_mod = sys.modules["data"]
    test_node_loader_dict = {}
    for case in test_case_names:
        test_node_loader_dict[case] = data_mod.load_multicase_graph_test_dataset(device)

    best_test_results = evaluate_test(
        best_model, test_node_loader_dict, test_case_names,
        node_feat_dim, edge_feat_dim, device, epoch=best_epoch,
    )

    # ---- Print final summary table --------------------------------------------
    if best_test_results:
        print("\n" + "=" * 100)
        print(f"  Best epoch: {best_epoch} | val_arc_slew_r2: {best_val_slew_r2:.4f} | "
              f"val_arc_delay_r2: {best_val_r2:.4f}")
        print("-" * 100)
        header = f"  {'Design':12s} | {'arc_slew_r2':>11s} | {'arc_delay_r2':>12s} | "
        header += f"{'path_r2':>8s} | {'path_mae':>10s} | {'path_max_err':>12s} | {'time':>5s}"
        print(header)
        print("-" * 100)
        cases = test_case_names if test_case_names else list(best_test_results.keys())
        avg = {"arc_slew_r2": [], "arc_delay_r2": [], "path_r2": [],
               "path_mae": [], "path_max_err": [], "infer_time": []}
        for case in cases:
            m = best_test_results.get(case, {})
            print(f"  {case:12s} | {m.get('arc_slew_r2', 0):11.4f} | {m.get('arc_delay_r2', 0):12.4f} | "
                  f"{m.get('path_r2', 0):8.4f} | {m.get('path_mae', 0):10.4f} | "
                  f"{m.get('path_max_err', 0):12.4f} | {m.get('infer_time', 0):5.2f}s")
            for k in avg:
                if k in m:
                    avg[k].append(m[k])
        print("-" * 100)
        if avg["arc_delay_r2"]:
            print(f"  {'Average':12s} | {sum(avg['arc_slew_r2'])/len(avg['arc_slew_r2']):11.4f} | "
                  f"{sum(avg['arc_delay_r2'])/len(avg['arc_delay_r2']):12.4f} | "
                  f"{sum(avg['path_r2'])/len(avg['path_r2']):8.4f} | "
                  f"{sum(avg['path_mae'])/len(avg['path_mae']):10.4f} | "
                  f"{sum(avg['path_max_err'])/len(avg['path_max_err']):12.4f} | "
                  f"{sum(avg['infer_time'])/len(avg['infer_time']):5.2f}s")
        print("=" * 100)


def parse_opt():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=str(CONFIG_PATH), help="config yaml path")
    parser.add_argument("--epochs", type=int, default=None, help="total train epoch (override config)")
    parser.add_argument("--batch-size", type=int, default=None, help="train batch size (override config)")
    parser.add_argument("--project", default=str(ROOT / _cfg("paths.project", "./runs/")), help="save root dir")
    parser.add_argument("--name", default="exp", help="experiment name")
    parser.add_argument("--train_cases", type=str, default=None,
                        help="train case list split by comma (override config)")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
