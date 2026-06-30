"""
Test evaluation for PBA-rGAT-Edge: run inference on test cases and report metrics.
"""
import os
import sys
import time
from pathlib import Path

import torch

# Ensure this directory is on sys.path for local imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model import r2_score, calculate_mae, max_diff

# Project root (relative to this file)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def evaluate_test(model, test_node_loader_dict, test_case_names,
                  node_feat_dim, edge_feat_dim, device, epoch=0):
    """Run inference on test cases and return per-case metrics dict."""
    results = {}
    model.eval()
    with torch.no_grad():
        for case in test_case_names:
            start_test = time.time()
            total_pred_slew = torch.tensor([], device=device)
            total_pred_delay = torch.tensor([], device=device)
            total_label_slew = torch.tensor([], device=device)
            total_label_delay = torch.tensor([], device=device)
            total_path_pred_delay = torch.tensor([], device=device)
            total_path_label_delay = torch.tensor([], device=device)

            node_loader = test_node_loader_dict[case]
            for node_data in node_loader:
                node_attr = node_data.x[:, :node_feat_dim].to(device)
                edge_index = node_data.edge_index
                edge_attr = node_data.edge_attr[:, :edge_feat_dim].to(device)
                slew, delay = model(node_attr, edge_index, edge_attr)

                total_pred_slew = torch.cat((total_pred_slew, slew), dim=0)
                total_pred_delay = torch.cat((total_pred_delay, delay), dim=0)
                total_label_slew = torch.cat((total_label_slew, node_data.label_slew.to(device)), dim=0)
                total_label_delay = torch.cat((total_label_delay, node_data.label_delay.to(device)), dim=0)

                path_pred = torch.tensor([delay.sum().item()], device=device)
                path_label = torch.tensor([node_data.label_delay.sum().item()], device=device)
                total_path_pred_delay = torch.cat((total_path_pred_delay, path_pred), dim=0)
                total_path_label_delay = torch.cat((total_path_label_delay, path_label), dim=0)

            infer_time = time.time() - start_test
            slew_r2 = r2_score(total_label_slew, total_pred_slew)
            delay_r2 = r2_score(total_label_delay, total_pred_delay)
            path_delay_r2 = r2_score(total_path_label_delay, total_path_pred_delay)
            path_delay_mae = calculate_mae(total_path_label_delay, total_path_pred_delay)
            delay_max_diff = max_diff(total_path_label_delay, total_path_pred_delay)

            test_save_dir = os.path.join(ROOT, "output/pbargatedge/test", case)
            os.makedirs(test_save_dir, exist_ok=True)
            torch.save({
                "delay_max_diff": delay_max_diff,
                "slew_r2": slew_r2,
                "delay_r2": delay_r2,
                "path_delay_r2": path_delay_r2,
                "path_delay_mae": path_delay_mae,
                "total_label_slew": total_label_slew,
                "total_pred_slew": total_pred_slew,
                "total_label_delay": total_label_delay,
                "total_pred_delay": total_pred_delay,
                "total_path_label_delay": total_path_label_delay,
                "total_path_pred_delay": total_path_pred_delay,
                "infer_time": infer_time,
            }, os.path.join(test_save_dir, f"epoch_{epoch}.pt"))

            print(f"  Test {case:12s} | {infer_time:.2f}s | "
                  f"arc_slew_r2 {slew_r2:.4f} | arc_delay_r2 {delay_r2:.4f} | "
                  f"path_r2 {path_delay_r2:.4f} | path_mae {path_delay_mae:10.4f} | "
                  f"path_max_err {delay_max_diff:.4f}", flush=True)

            results[case] = {
                "arc_slew_r2": slew_r2, "arc_delay_r2": delay_r2,
                "path_r2": path_delay_r2, "path_mae": path_delay_mae,
                "path_max_err": delay_max_diff, "infer_time": infer_time,
            }
    return results
