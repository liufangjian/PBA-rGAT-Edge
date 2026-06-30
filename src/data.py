"""
Synthetic data generation (EDA STA physical law compliant) and data loading utilities.
"""
import copy
import os
import pickle
import random
import time

import numpy as np
import torch
from torch_geometric.data import Data


# ===========================================================================
# Synthetic data generator following EDA STA physical laws (dual-pass Elmore)
# Features defined as in paper Table 1:
#   Node features (4-dim): [slew_type, max_rise_transition_time, max_fall_transition_time,
#                           direction_of_signal]
#   Edge features (5-dim): [GBA_delay, arc_type, max_total_resistance,
#                           ccs_total_max_rise_cap, ccs_total_max_fall_cap]
# Labels: PBA-quality slew and delay (computed via Elmore delay + slew propagation)
# Task:  Predict PBA (accurate) node-level timing from GBA (pessimistic) edge features
# Design:
#   - Pass 1 (PBA): nominal internal params -> PBA delay/slew (labels)
#   - Pass 2 (GBA): inflated internal params -> GBA slew (node input features)
#   - Edge features: computed from physical params + GBA pessimism multiplier
#   - Node features: GBA slew directly populates max_rise/fall_transition_time;
#                    direction_of_signal derived from graph topology
# ===========================================================================
class SyntheticGraphGenerator:
    """Generate synthetic timing graph data aligned with paper Table 1 definitions"""
    def __init__(
        self,
        node_feat_dim: int = 4,
        edge_feat_dim: int = 5,
        min_nodes: int = 20,
        max_nodes: int = 100,
        min_layers: int = 3,
        max_layers: int = 6,
        max_fan_in: int = 3,
        noise_ratio: float = 0.005,
        device: str = "cpu"
    ):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.max_fan_in = max_fan_in
        self.noise_ratio = noise_ratio
        self.device = device

    # ------------------------------------------------------------------
    # Internal physical parameters (for Elmore delay computation, NOT exposed to model)
    # ------------------------------------------------------------------
    def _generate_internal_params(self, num_nodes: int, num_edges: int):
        """Generate internal physical parameters for Elmore delay computation"""
        return {
            'c_load': torch.empty(num_nodes, device=self.device).uniform_(1.0, 3.0),
            'r_drive': torch.empty(num_nodes, device=self.device).uniform_(0.5, 2.0),
            't_intrinsic': torch.empty(num_nodes, device=self.device).uniform_(0.1, 0.5),
            'k_proc': torch.empty(num_nodes, device=self.device).uniform_(0.9, 1.1),
            'wire_len': torch.empty(num_edges, device=self.device).uniform_(0.3, 3.0),
            'r_per_len': torch.empty(num_edges, device=self.device).uniform_(0.05, 0.2),
            'c_per_len': torch.empty(num_edges, device=self.device).uniform_(0.02, 0.1),
            'width_factor': torch.empty(num_edges, device=self.device).uniform_(0.8, 1.5),
            'k_couple': torch.empty(num_edges, device=self.device).uniform_(0.0, 0.5),
        }

    # ------------------------------------------------------------------
    # Graph structure generation
    # ------------------------------------------------------------------
    def _generate_dag_edges(self, num_nodes: int):
        """
        Generate DAG edge index simulating circuit signal flow.
        Ensures every node has at least one incoming edge, connected by topological layer.
        Returns: edge_index [2, num_edges], layer_ids [num_nodes]
        """
        n_layers = random.randint(self.min_layers, self.max_layers)
        layer_ids = torch.randint(0, n_layers, (num_nodes,), device=self.device)

        # Ensure at least one node per layer
        for l in range(n_layers):
            if not (layer_ids == l).any():
                idx = random.randint(0, num_nodes - 1)
                layer_ids[idx] = l

        src_list = []
        dst_list = []

        for j in range(num_nodes):
            j_layer = layer_ids[j].item()
            if j_layer == 0:
                src_list.append(j)
                dst_list.append(j)
            else:
                prev_nodes = torch.where(layer_ids == j_layer - 1)[0]
                fan_in = min(random.randint(1, self.max_fan_in), len(prev_nodes))
                selected = prev_nodes[torch.randperm(len(prev_nodes))[:fan_in]]
                for src_node in selected:
                    src_list.append(src_node.item())
                    dst_list.append(j)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)
        return edge_index, layer_ids

    # ------------------------------------------------------------------
    # Node features (paper Table 1) - assembled from physically computed values
    # ------------------------------------------------------------------
    def _generate_node_features(self, gba_slew: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Assemble node features from physically computed values (paper Table 1).

        [slew_type, max_rise_transition_time, max_fall_transition_time, direction_of_signal]

        - max_rise_transition_time / max_fall_transition_time: GBA-pessimistic slew
        - direction_of_signal: from graph topology (0=input, 1=output)
        """
        num_nodes = gba_slew.size(0)
        # Note: in synthetic data slew_type is derived from direction (pin I/O type),
        # making it a duplicate. In real STA data, slew_type (rise/fall transition)
        # and direction_of_signal (input/output pin) would be independent.
        slew_type = direction.clone()
        max_rise_trans = gba_slew * torch.empty(num_nodes, device=self.device).uniform_(0.998, 1.002)
        max_fall_trans = gba_slew * torch.empty(num_nodes, device=self.device).uniform_(0.998, 1.002)
        return torch.stack([slew_type, max_rise_trans, max_fall_trans, direction], dim=-1)

    # ------------------------------------------------------------------
    # Elmore propagation engine (used by both PBA and GBA passes)
    # ------------------------------------------------------------------
    def _propagate_timing(self, internal, edge_index, layer_ids):
        """
        Propagate delay and slew through the DAG using Elmore model.

        Args:
            internal: dict of physical parameters (c_load, r_drive, etc.)
            edge_index: [2, num_edges] graph connectivity
            layer_ids: [num_nodes] topological layer assignment

        Returns:
            delay [num_nodes], slew [num_nodes]: per-node timing values
        """
        c_load = internal['c_load']
        r_drive = internal['r_drive']
        t_intrinsic = internal['t_intrinsic']
        k_proc = internal['k_proc']
        wire_len = internal['wire_len']
        r_per_len = internal['r_per_len']
        c_per_len = internal['c_per_len']
        width_factor = internal['width_factor']
        k_couple = internal['k_couple']

        num_nodes = c_load.size(0)
        delay = torch.zeros(num_nodes, device=self.device)
        slew = torch.zeros(num_nodes, device=self.device)

        max_layer = layer_ids.max().item()
        for l in range(int(max_layer) + 1):
            layer_nodes = torch.where(layer_ids == l)[0]
            for j in layer_nodes:
                j = j.item()
                if l == 0:
                    delay[j] = t_intrinsic[j] * k_proc[j]
                    slew[j] = 0.8 * k_proc[j]
                else:
                    edge_mask = edge_index[1] == j
                    edge_indices = torch.where(edge_mask)[0]
                    cand_delays, cand_slews = [], []

                    for eid in edge_indices:
                        i = edge_index[0, eid].item()
                        L = wire_len[eid]
                        r = r_per_len[eid]
                        c = c_per_len[eid]
                        w = width_factor[eid]
                        kc = k_couple[eid]

                        R_wire = r * L / w
                        C_wire = c * L * (1 + kc)

                        # Elmore wire delay
                        d_wire = R_wire * C_wire * 0.5 + R_wire * c_load[j]
                        d_total = delay[i] + d_wire + t_intrinsic[j] * k_proc[j]

                        # Slew degradation
                        C_total = C_wire + c_load[j]
                        slew_degrade = r_drive[j] * C_total * k_proc[j] * 0.3
                        s_total = torch.sqrt(slew[i] ** 2 + slew_degrade ** 2)

                        cand_delays.append(d_total)
                        cand_slews.append(s_total)

                    # Worst-case path (STA rule)
                    delay[j] = torch.stack(cand_delays).max()
                    slew[j] = torch.stack(cand_slews).max()

        return delay, slew

    # ------------------------------------------------------------------
    # GBA pessimism: inflate wire parameters to simulate GBA analysis
    # ------------------------------------------------------------------
    def _apply_gba_pessimism(self, internal):
        """
        Create pessimistic (GBA-style) internal params from nominal ones.

        Inflates wire length, coupling, and drive resistance to simulate
        the pessimism inherent in Graph-Based Analysis.
        """
        gba = {}
        gba['c_load'] = internal['c_load'].clone()
        gba['r_drive'] = internal['r_drive'] * 1.12
        gba['t_intrinsic'] = internal['t_intrinsic'] * 1.08
        gba['k_proc'] = internal['k_proc'] * 1.05
        gba['wire_len'] = internal['wire_len'] * 1.15
        gba['r_per_len'] = internal['r_per_len'] * 1.10
        gba['c_per_len'] = internal['c_per_len'] * 1.10
        gba['width_factor'] = internal['width_factor'] / 1.05
        gba['k_couple'] = internal['k_couple'] + 0.08
        return gba

    # ------------------------------------------------------------------
    # PBA labels + GBA features (dual-pass Elmore propagation)
    # ------------------------------------------------------------------
    def _compute_pba_labels_and_gba_features(self, num_nodes, edge_index, internal, layer_ids):
        """
        Dual-pass Elmore computation producing physically consistent features.

        Pass 1 (PBA): nominal params -> PBA-quality delay/slew (labels)
        Pass 2 (GBA): pessimistic params -> GBA-quality delay/slew (node input features)

        Edge features: derived from physical + GBA pessimism.
        Direction: derived from graph topology.
        """
        # ---- Pass 1: PBA-accurate (nominal parameters) ----
        PBA_delay, PBA_slew = self._propagate_timing(internal, edge_index, layer_ids)

        # Add process noise to labels
        PBA_delay += torch.randn_like(PBA_delay) * PBA_delay * self.noise_ratio
        PBA_slew += torch.randn_like(PBA_slew) * PBA_slew * self.noise_ratio
        PBA_delay = torch.abs(PBA_delay)
        PBA_slew = torch.abs(PBA_slew)

        # ---- Pass 2: GBA-pessimistic (inflated parameters) ----
        gba_internal = self._apply_gba_pessimism(internal)
        _, GBA_slew = self._propagate_timing(gba_internal, edge_index, layer_ids)

        # GBA delay for edge features: per-edge wire delay with pessimism
        c_load = internal['c_load']
        wire_len = internal['wire_len']
        r_per_len = internal['r_per_len']
        c_per_len = internal['c_per_len']
        width_factor = internal['width_factor']
        k_couple = internal['k_couple']

        num_edges = edge_index.size(1)
        true_edge_delays = torch.zeros(num_edges, device=self.device)
        for eid in range(num_edges):
            j = edge_index[1, eid].item()
            L = wire_len[eid]
            r = r_per_len[eid]
            c = c_per_len[eid]
            w = width_factor[eid]
            kc = k_couple[eid]
            R_wire = r * L / w
            C_wire = c * L * (1 + kc)
            d_wire = R_wire * C_wire * 0.5 + R_wire * c_load[j]
            true_edge_delays[eid] = d_wire

        # GBA_delay: pessimistic wire delay (fixed 15% pessimism per graph)
        GBA_delay = true_edge_delays * 1.15

        # Edge features in paper Table 1 format
        arc_type = torch.ones(num_edges, device=self.device).float()
        total_R = r_per_len * wire_len / width_factor
        rise_cap = c_per_len * wire_len
        fall_cap = rise_cap * torch.empty(num_edges, device=self.device).uniform_(0.998, 1.002)
        edge_attrs = torch.stack([GBA_delay, arc_type, total_R, rise_cap, fall_cap], dim=-1)

        # Direction from graph topology: 0=input (layer 0), 1=output (others)
        direction = (layer_ids > 0).float()

        return PBA_slew, PBA_delay, edge_attrs, GBA_slew, direction

    # ------------------------------------------------------------------
    # Data generation entry points
    # ------------------------------------------------------------------
    def generate_single_graph(self):
        """Generate a single training graph with fully physical features."""
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        edge_index, layer_ids = self._generate_dag_edges(num_nodes)
        num_edges = edge_index.size(1)

        internal = self._generate_internal_params(num_nodes, num_edges)
        PBA_slew, PBA_delay, edge_attrs, GBA_slew, direction = \
            self._compute_pba_labels_and_gba_features(
                num_nodes, edge_index, internal, layer_ids
            )

        node_attrs = self._generate_node_features(GBA_slew, direction)

        graph_data = {
            "edge_aggregator_data": {
                "node_attrs": node_attrs,
                "edge_indexs": edge_index,
                "edge_attrs": edge_attrs
            },
            "node_aggregator_data": {
                "node_attrs": node_attrs,
                "node_indexs": edge_index,
                "edge_attrs": edge_attrs
            },
            "pba_net_edge_slew": PBA_slew,
            "pba_net_edge_delay": PBA_delay
        }
        return graph_data

    def generate_dataset(self, num_samples: int = 100):
        """Generate a list of dataset samples."""
        return [self.generate_single_graph() for _ in range(num_samples)]

    def generate_pyg_test_graph(self):
        """Generate a PyG test graph with fully physical features."""
        num_nodes = random.randint(20, 80)
        edge_index, layer_ids = self._generate_dag_edges(num_nodes)
        num_edges = edge_index.size(1)

        internal = self._generate_internal_params(num_nodes, num_edges)
        label_slew, label_delay, edge_attr, GBA_slew, direction = \
            self._compute_pba_labels_and_gba_features(
                num_nodes, edge_index, internal, layer_ids
            )

        x = self._generate_node_features(GBA_slew, direction)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            label_slew=label_slew,
            label_delay=label_delay
        ).to(self.device)


# ===========================================================================
# IO Tools
# ===========================================================================
def load_yaml(file_path: str):
    """Load and return YAML file contents, or None if missing."""
    import yaml
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_graph(path: str):
    """Load a pickled graph object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_from_pt(filename: str, device: str = "cpu"):
    start = time.time()
    data = torch.load(filename, map_location=device, weights_only=False)
    return data, time.time() - start


def load_from_pickle(filename: str):
    """Load a pickle file from disk."""
    start = time.time()
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data, time.time() - start


# ===========================================================================
# Data Loader (synthetic data generation)
# ===========================================================================
def sample_dataset(data_list, sampling_ratio: float, device: str,
                   node_feat_dim: int = 4, edge_feat_dim: int = 5):
    """Convert a list of graph dicts into an 8-tuple of tensors, optionally subsampling."""
    edge_aggregator_edge_indexs_list = []
    edge_aggregator_node_attrs_list = []
    edge_aggregator_edge_attrs_list = []
    node_aggregator_node_indexs_list = []
    node_aggregator_node_attrs_list = []
    node_aggregator_edge_attrs_list = []
    pba_net_edge_slew_list = []
    pba_net_edge_delay_list = []

    for data in data_list:
        edge_data = data["edge_aggregator_data"]
        node_data = data["node_aggregator_data"]

        # Slice training data to fixed feature dimensions
        node_x = edge_data["node_attrs"][:, :node_feat_dim].to(device)
        edge_x = edge_data["edge_attrs"][:, :edge_feat_dim].to(device)

        edge_aggregator_edge_indexs_list.append(edge_data["edge_indexs"])
        edge_aggregator_node_attrs_list.append(node_x)
        edge_aggregator_edge_attrs_list.append(edge_x)

        node_aggregator_node_indexs_list.append(node_data["node_indexs"])
        node_aggregator_node_attrs_list.append(node_data["node_attrs"][:, :node_feat_dim].to(device))
        node_aggregator_edge_attrs_list.append(node_data["edge_attrs"][:, :edge_feat_dim].to(device))

        pba_net_edge_slew_list.append(data["pba_net_edge_slew"].to(device))
        pba_net_edge_delay_list.append(data["pba_net_edge_delay"].to(device))

    if sampling_ratio == 1.0:
        edge_aggregator_edge_indexs_list = [t.long().to(device) for t in edge_aggregator_edge_indexs_list]
        dataset = (
            edge_aggregator_edge_indexs_list,
            edge_aggregator_node_attrs_list,
            edge_aggregator_edge_attrs_list,
            node_aggregator_node_indexs_list,
            node_aggregator_node_attrs_list,
            node_aggregator_edge_attrs_list,
            pba_net_edge_slew_list,
            pba_net_edge_delay_list,
        )
        data_num = len(edge_aggregator_edge_indexs_list)
    else:
        n = len(edge_aggregator_node_attrs_list)
        k = max(1, int(n * sampling_ratio))
        sampled_indices = random.sample(range(n), k)

        def _gather(lst, indices):
            return [lst[i] for i in indices]

        edge_aggregator_edge_indexs_split = [t.long().to(device) for t in
                                             _gather(edge_aggregator_edge_indexs_list, sampled_indices)]
        edge_aggregator_node_attrs_split = _gather(edge_aggregator_node_attrs_list, sampled_indices)
        edge_aggregator_edge_attrs_split = _gather(edge_aggregator_edge_attrs_list, sampled_indices)

        node_aggregator_node_indexs_split = [t.long().to(device) for t in
                                             _gather(node_aggregator_node_indexs_list, sampled_indices)]
        node_aggregator_node_attrs_split = _gather(node_aggregator_node_attrs_list, sampled_indices)
        node_aggregator_edge_attrs_split = _gather(node_aggregator_edge_attrs_list, sampled_indices)

        dataset = (
            edge_aggregator_edge_indexs_split,
            edge_aggregator_node_attrs_split,
            edge_aggregator_edge_attrs_split,
            node_aggregator_node_indexs_split,
            node_aggregator_node_attrs_split,
            node_aggregator_edge_attrs_split,
            _gather(pba_net_edge_slew_list, sampled_indices),
            _gather(pba_net_edge_delay_list, sampled_indices),
        )
        data_num = len(sampled_indices)
    return dataset, data_num


def load_multicase_graph_test_dataset(device: str = "cpu"):
    """Generate synthetic test graphs using EDA physical laws"""
    generator = SyntheticGraphGenerator(device=device)
    all_test_dataset = []
    for _ in range(50):
        all_test_dataset.append(generator.generate_pyg_test_graph())
    return all_test_dataset


def load_multicase_dataset(cases_path: str, case_names, train_flag: bool, val_flag: bool,
                           device: str, batch_size: int, sampling_ratio: float = 1.0,
                           num_train_samples: int = 500, num_val_samples: int = 100,
                           node_feat_dim: int = 4, edge_feat_dim: int = 5):
    """Generate synthetic train/val datasets using EDA physical laws"""
    generator = SyntheticGraphGenerator(device=device)

    train_temp = generator.generate_dataset(num_samples=num_train_samples)
    val_temp = generator.generate_dataset(num_samples=num_val_samples)

    train_list = train_temp
    val_list = val_temp

    total_train_num = 0
    total_val_num = 0

    def _empty_lists():
        return [[] for _ in range(8)]

    (edge_aggregator_edge_indexs_train, edge_aggregator_node_attrs_train, edge_aggregator_edge_attrs_train,
     node_aggregator_node_indexs_train, node_aggregator_node_attrs_train, node_aggregator_edge_attrs_train,
     pba_net_edge_slew_train, pba_net_edge_delay_train) = _empty_lists()

    (edge_aggregator_edge_indexs_val, edge_aggregator_node_attrs_val, edge_aggregator_edge_attrs_val,
     node_aggregator_node_indexs_val, node_aggregator_node_attrs_val, node_aggregator_edge_attrs_val,
     pba_net_edge_slew_val, pba_net_edge_delay_val) = _empty_lists()

    if train_flag:
        train_data, batch_num_train = sample_dataset(train_list, sampling_ratio, device,
                                                      node_feat_dim, edge_feat_dim)
        for i in range(8):
            lst = [edge_aggregator_edge_indexs_train, edge_aggregator_node_attrs_train,
                   edge_aggregator_edge_attrs_train, node_aggregator_node_indexs_train,
                   node_aggregator_node_attrs_train, node_aggregator_edge_attrs_train,
                   pba_net_edge_slew_train, pba_net_edge_delay_train][i]
            lst.extend(train_data[i])
        total_train_num += batch_num_train

    if val_flag:
        val_data, batch_num_val = sample_dataset(val_list, 1.0, device,
                                                  node_feat_dim, edge_feat_dim)
        for i in range(8):
            lst = [edge_aggregator_edge_indexs_val, edge_aggregator_node_attrs_val,
                   edge_aggregator_edge_attrs_val, node_aggregator_node_indexs_val,
                   node_aggregator_node_attrs_val, node_aggregator_edge_attrs_val,
                   pba_net_edge_slew_val, pba_net_edge_delay_val][i]
            lst.extend(val_data[i])
        total_val_num += batch_num_val

    total_train = (
        edge_aggregator_edge_indexs_train,
        edge_aggregator_node_attrs_train,
        edge_aggregator_edge_attrs_train,
        node_aggregator_node_indexs_train,
        node_aggregator_node_attrs_train,
        node_aggregator_edge_attrs_train,
        pba_net_edge_slew_train,
        pba_net_edge_delay_train,
    )
    total_val = (
        edge_aggregator_edge_indexs_val,
        edge_aggregator_node_attrs_val,
        edge_aggregator_edge_attrs_val,
        node_aggregator_node_indexs_val,
        node_aggregator_node_attrs_val,
        node_aggregator_edge_attrs_val,
        pba_net_edge_slew_val,
        pba_net_edge_delay_val,
    )
    return total_train, total_train_num, total_val, total_val_num
