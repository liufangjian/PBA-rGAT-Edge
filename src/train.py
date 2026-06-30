"""
PBA-rGAT-Edge training pipeline: config loading, training loop, and entry point.
"""
import copy
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Load local modules by file path (bypasses sys.path issues)
# ---------------------------------------------------------------------------
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_module(module_name, filename):
    """Load a Python file as a module by absolute path."""
    import importlib.util
    filepath = os.path.join(_MODULES_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order (test -> depends on model)
_model = _load_local_module("model", "model.py")
_data = _load_local_module("data", "data.py")

PBARGATEdge = _model.PBARGATEdge
r2_score = _model.r2_score
calculate_mae = _model.calculate_mae
max_diff = _model.max_diff
load_multicase_dataset = _data.load_multicase_dataset

# ---------------------------------------------------------------------------
# Suppress NVML warning
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message=r".*Can't initialize NVML.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

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
# Global Constants (populated from YAML config)
# ---------------------------------------------------------------------------
node_feat_dim = _cfg("model.node_feat_dim", 4)
edge_feat_dim = _cfg("model.edge_feat_dim", 5)
loss_coefficient_slew = _cfg("training.loss_coefficient.slew", 0.05)
loss_coefficient_delay = _cfg("training.loss_coefficient.delay", 0.95)


# ---------------------------------------------------------------------------
# Early Stopper
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Early stopping callback that tracks validation loss improvement."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ===========================================================================
# Training Pipeline
# ===========================================================================
def run_pipeline(opt, device: str):
    """Main training loop with synthetic data, validation, and test inference."""
    save_dir = Path(opt.save_dir)

    # CLI args override config defaults
    epochs = opt.epochs if opt.epochs is not None else _cfg("training.epochs", 30)
    batch_size = opt.batch_size if opt.batch_size is not None else _cfg("training.batch_size", 64)
    train_cases_str = opt.train_cases if opt.train_cases is not None else ",".join(_cfg("data.train_cases", ["design1"]))
    sampling_ratio = _cfg("data.sampling_ratio", 1.0)

    current_epoch = 0
    train_case_names = [x.strip() for x in train_cases_str.split(",")]
    lr = _cfg("training.lr", 1e-3)
    gat_layer_num = _cfg("model.gat_layer_num", 3)

    num_train_samples = _cfg("data.num_train_samples", 500)
    num_val_samples = _cfg("data.num_val_samples", 100)

    print(f">> Config: epochs={epochs} batch_size={batch_size} lr={lr} device={device}")
    print(f">> Train cases: {train_case_names}")
    print(">> Generating training/validation data ... ", end="", flush=True)

    save_path = os.path.join(ROOT, save_dir)
    os.makedirs(save_path, exist_ok=True)

    train_data, batch_num_train, val_data, batch_num_val = load_multicase_dataset(
        "", train_case_names, True, True, device, batch_size, sampling_ratio,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
    )

    print(f"done ({batch_num_train} train / {batch_num_val} val graphs).")

    model = PBARGATEdge(node_feat_dim, edge_feat_dim, gat_layer_num).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f">> Model: PBARGATEdge | params: {n_params:,} | device: {device}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(_cfg("optimizer.weight_decay", 1e-5)),
    )
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode=_cfg("scheduler.mode", "min"),
        factor=_cfg("scheduler.factor", 0.5),
        patience=_cfg("scheduler.patience", 3),
        min_lr=float(_cfg("scheduler.min_lr", 1e-6)),
        threshold=float(_cfg("scheduler.threshold", 1e-4)),
    )

    best_val_r2 = 0.0
    best_val_slew_r2 = 0.0
    best_epoch = 0
    best_test_results = {}

    loss_fn = nn.HuberLoss()
    early_stopper = EarlyStopper(
        patience=_cfg("early_stopper.patience", 10),
        min_delta=float(_cfg("early_stopper.min_delta", 0.0001)),
    )

    (edge_aggregator_edge_indexs_train, edge_aggregator_node_attrs_train,
     edge_aggregator_edge_attrs_train, *_,
     pba_net_edge_slew_train, pba_net_edge_delay_train) = train_data

    total_val = val_data
    if total_val is not None:
        (edge_aggregator_edge_indexs_val, edge_aggregator_node_attrs_val,
         edge_aggregator_edge_attrs_val, *_,
         pba_net_edge_slew_val, pba_net_edge_delay_val) = total_val

    print(">> Training started.")
    start_time = time.time()

    for epoch in range(current_epoch + 1, epochs + 1):
        running_loss = 0.0
        model.train()

        start_train = time.time()
        for batch_idx in range(batch_num_train):
            optimizer.zero_grad()
            pred_slew, pred_delay = model(
                edge_aggregator_node_attrs_train[batch_idx],
                edge_aggregator_edge_indexs_train[batch_idx],
                edge_aggregator_edge_attrs_train[batch_idx],
            )
            loss_slew = loss_fn(pred_slew, pba_net_edge_slew_train[batch_idx])
            loss_delay = loss_fn(pred_delay, pba_net_edge_delay_train[batch_idx])
            train_total_loss = loss_coefficient_slew * loss_slew + loss_coefficient_delay * loss_delay
            running_loss += train_total_loss

            train_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _cfg("training.clip_grad_norm", 1.0))
            optimizer.step()

        train_time = time.time() - start_train
        train_loss = running_loss / batch_num_train

        total_preds_slew_val = []
        total_preds_delay_val = []
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for batch_idx in range(batch_num_val):
                pred_slew, pred_delay = model(
                    edge_aggregator_node_attrs_val[batch_idx],
                    edge_aggregator_edge_indexs_val[batch_idx],
                    edge_aggregator_edge_attrs_val[batch_idx],
                )
                total_preds_slew_val.append(copy.deepcopy(pred_slew))
                total_preds_delay_val.append(copy.deepcopy(pred_delay))
                loss_slew_val = loss_fn(pred_slew, pba_net_edge_slew_val[batch_idx])
                loss_delay_val = loss_fn(pred_delay, pba_net_edge_delay_val[batch_idx])
                val_total_loss = loss_coefficient_slew * loss_slew_val + loss_coefficient_delay * loss_delay_val
                running_loss_val += val_total_loss

        val_loss = running_loss_val / batch_num_val

        if total_preds_slew_val:
            all_preds_slew_val = torch.cat(total_preds_slew_val, dim=-1)
            all_preds_delay_val = torch.cat(total_preds_delay_val, dim=-1)
            slew_true = torch.cat(pba_net_edge_slew_val).view(-1)
            delay_true = torch.cat(pba_net_edge_delay_val).view(-1)
            accuracy_slew_val = r2_score(slew_true, all_preds_slew_val)
            accuracy_delay_val = r2_score(delay_true, all_preds_delay_val)
        else:
            accuracy_slew_val = 0.0
            accuracy_delay_val = 0.0

        scheduler_plateau.step(val_loss)
        current_lr = scheduler_plateau.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{epochs} | {train_time:.2f}s | "
              f"val_loss {val_loss.item():.6f} | "
              f"arc_slew_r2 {accuracy_slew_val:.4f} | arc_delay_r2 {accuracy_delay_val:.4f} | "
              f"lr {current_lr:.2e}", flush=True)

        dump_mae_path = os.path.join(ROOT, "output/rgat/train")
        os.makedirs(dump_mae_path, exist_ok=True)
        torch.save({
            "train_time": train_time,
            "loss_val": val_loss.item(),
            "current_lr": current_lr,
            "slew_r2_val": accuracy_slew_val,
            "delay_r2_val": accuracy_delay_val,
        }, os.path.join(dump_mae_path, f"epoch_{epoch}.pt"))

        if early_stopper.step(train_loss.item()):
            print(f"  !! Early stop at epoch {epoch}")
            break

        model_save_path = os.path.join(ROOT, "runs/rgat_1108")
        if best_val_r2 < accuracy_delay_val:
            best_val_r2 = accuracy_delay_val
            best_val_slew_r2 = accuracy_slew_val
            best_epoch = epoch
            os.makedirs(model_save_path, exist_ok=True)
            best_model_path = os.path.join(model_save_path, "rgat_best.pt")
            torch.save(model, best_model_path)

    exec_time = time.time() - start_time
    print(f">> Training finished. Total time: {exec_time:.2f}s")

    return model, best_val_slew_r2, best_val_r2, best_epoch, model_save_path, epochs
