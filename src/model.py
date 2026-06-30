"""
PBA-rGAT-Edge model architecture and evaluation metrics (paper Sections 4.3, 4.4).
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (derived from config: 2 * node_feat_dim + edge_feat_dim = 2*4 + 5 = 13)
# ---------------------------------------------------------------------------
ATT_INPUT_DIM = 13


# ===========================================================================
# Model Modules (paper Section 4.3)
# ===========================================================================
class MLPPredictor(nn.Module):
    """MLP predictor head for per-node timing prediction.

    Maps fused multi-layer features to a scalar output (slew or delay).
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        hidden_dim = 64
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class EdgeAwareAttention(MessagePassing):
    """Edge-aware attention aggregation from paper Section 4.3.

    Computes attention scores using [x_i || x_j || e_ij] (Eq. 2),
    normalizes via softmax (Eq. 3), and aggregates g_i = sum(alpha * [x_j || e_ij]) (Eq. 4).
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int):
        super().__init__(node_dim=0, aggr="add")
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        att_input_dim = ATT_INPUT_DIM
        self.f_att = nn.Sequential(
            nn.Linear(att_input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1),
        )
        self._dst_index = None

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_index = edge_index.long()
        if edge_index.size(0) != 2:
            edge_index = edge_index.t().contiguous()
        self._dst_index = edge_index[1]
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr)
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x_i = x_i[..., :self.node_feat_dim]
        x_j = x_j[..., :self.node_feat_dim]
        edge_attr = edge_attr[..., :self.edge_feat_dim]
        alpha_combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha_combined = self.f_att(alpha_combined).squeeze(-1)
        alpha = softmax(alpha_combined, self._dst_index, ptr, size_i).unsqueeze(-1)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, edge_attr: Tensor) -> Tensor:
        node_combined = torch.cat([x_j, edge_attr], dim=-1)
        return node_combined * alpha

    def aggregate(self, inputs: Tensor, index: Tensor) -> Tensor:
        return super().aggregate(inputs, index)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class ResidualEncoder(nn.Module):
    """Residual encoding from paper Section 4.3, Eqs. 4-5.

    Computes x̃^d_i = σ(LN(W_enc · g^d_i + b_enc))  (Eq. 4)
    then x^d_i = x^(d-1)_i + x̃^d_i                     (Eq. 5)
    where g^d_i = Σ α^d_ij [x^(d-1)_j || e_ij] is the edge-aware aggregate,
    W_enc ∈ ℝ^{D_n × (D_n + D_e)}, LN = LayerNorm, σ = LeakyReLU.
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, node_feat_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_prev: Tensor, g_i: Tensor) -> Tensor:
        return x_prev + self.node_proj(g_i)


class SlewDelayPredictor(nn.Module):
    """Dual-head MLP predictor for simultaneous slew and delay regression.

    Uses shared architecture with independent heads for slew and delay,
    initialized via Xavier uniform.
    """

    def __init__(self, fused_dim: int):
        super().__init__()
        self.slew_predictor = MLPPredictor(input_dim=fused_dim)
        self.delay_predictor = MLPPredictor(input_dim=fused_dim)
        self._init_weight()

    def _init_weight(self):
        for predictor in (self.slew_predictor, self.delay_predictor):
            for m in predictor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, fused_features: Tensor):
        slew = self.slew_predictor(fused_features).squeeze(-1)
        delay = self.delay_predictor(fused_features).squeeze(-1)
        return slew, delay


class ResAttEdgeLayer(nn.Module):
    """Single ResAtt-Edge layer combining attention aggregation and residual encoding."""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int):
        super().__init__()
        self.node_attention = EdgeAwareAttention(node_feat_dim, edge_feat_dim)
        self.residual_encoder = ResidualEncoder(node_feat_dim, edge_feat_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        g_i = self.node_attention(x, edge_index, edge_attr)
        return self.residual_encoder(x, g_i)


class PBARGATEdge(nn.Module):
    """PBA-rGAT-Edge model: multi-layer ResAtt-Edge with multi-level feature fusion.

    Stacks multiple ResAttEdgeLayer blocks, concatenates all layer outputs (Eq. 6),
    and predicts slew and delay via dual-head MLP.
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.resatt_layers = nn.ModuleList(
            [ResAttEdgeLayer(node_feat_dim, edge_feat_dim) for _ in range(num_layers)]
        )
        self.predict_layers = SlewDelayPredictor(node_feat_dim * num_layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x = x[..., :self.node_feat_dim]
        edge_attr = edge_attr[..., :self.edge_feat_dim]
        edge_index = edge_index.long()
        if edge_index.size(0) != 2:
            edge_index = edge_index.t().contiguous()

        layer_outputs = []
        for layer in self.resatt_layers:
            x = layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
        fused = torch.cat(layer_outputs, dim=-1)
        return self.predict_layers(fused)


# ===========================================================================
# Model utility
# ===========================================================================
def adjust_bn_momentum(model: nn.Module, epoch: int, max_epochs: int) -> None:
    """Gradually increase BatchNorm momentum over training epochs."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 0.1 + 0.9 * (epoch / max_epochs)


# ===========================================================================
# Evaluation Metrics
# ===========================================================================
def r2_score(y_true: Tensor, y_pred: Tensor, eps: float = 1e-8) -> float:
    """Compute coefficient of determination (R^2) score."""
    y_mean = torch.mean(y_true)
    sst = torch.sum((y_true - y_mean) ** 2)
    ssr = torch.sum((y_true - y_pred) ** 2)
    return (1.0 - (ssr / (sst + eps))).item()


def calculate_mae(y_true: Tensor, y_pred: Tensor) -> float:
    """Compute mean absolute error."""
    return nn.L1Loss()(y_pred, y_true).item()


def max_diff(y_true: Tensor, y_pred: Tensor) -> float:
    """Compute maximum absolute difference between predictions and labels."""
    return torch.max(torch.abs(y_pred - y_true)).item()


def mae_numpy(y_true: Tensor, y_pred: Tensor, dim: int = -1) -> Tensor:
    """Compute mean absolute error along a given dimension."""
    return torch.mean(torch.abs(y_true - y_pred), dim=dim)
