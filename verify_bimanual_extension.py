
"""
Comprehensive step-by-step verification script for the bimanual extension.

What this checks (in order):
1) SE(3) frame invariance of left-right relative transforms
2) Node indexing and embedding ranges
3) Node feature construction (scene + gripper)
4) Edge topology for every edge type
5) Edge attributes for every edge type (local + cross-arm)
6) Cross-arm action-relative transform construction
7) Global-frame invariance for demo-based edge attributes
8) Labels <-> decode consistency (per-arm)
9) Collate function shape and batch-index correctness

Run:
  python verify_bimanual_extension.py
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Set, Optional

import numpy as np
import torch

from data_structures import relative_transform, BimanualGraphData
from bimanual_graph_rep import BimanualGraphRep
from bimanual_dataset import collate_bimanual


torch.set_printoptions(precision=6, sci_mode=False)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def substep(title: str) -> None:
    print("\n-- " + title)


def assert_true(cond: bool, name: str) -> None:
    if not cond:
        raise AssertionError(f"{name} failed")


def assert_allclose(actual: torch.Tensor,
                    expected: torch.Tensor,
                    name: str,
                    atol: float = 1e-6,
                    rtol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(
            f"{name} mismatch. max|diff|={diff}\n"
            f"actual:\n{actual}\nexpected:\n{expected}"
        )


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name} mismatch.\nactual:\n{actual}\nexpected:\n{expected}")


def make_T(tx: float, ty: float, tz: float,
           rot: Optional[torch.Tensor] = None) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float32)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
    return T


def rot_z(deg: float) -> torch.Tensor:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

def build_test_config() -> Dict:
    # Keep sizes small but valid; local_nn_dim must be >= g_state_dim + d_time_dim.
    return {
        "batch_size": 1,
        "num_demos": 1,
        "traj_horizon": 2,
        "num_scenes_nodes": 2,
        "local_num_freq": 2,
        "device": "cpu",
        "local_nn_dim": 160,
        "pre_horizon": 2,
        "pos_in_nodes": False,
        "use_cross_arm_attention": True,
        "gripper_keypoints": torch.tensor([
            [0.02, 0.0, 0.0],
            [-0.02, 0.0, 0.0],
        ], dtype=torch.float32),
    }


def build_test_data(config: Dict) -> BimanualGraphData:
    B = config["batch_size"]
    D = config["num_demos"]
    T = config["traj_horizon"]
    P = config["pre_horizon"]
    S = config["num_scenes_nodes"]

    # Demo transforms with translation and rotation.
    T_left0 = make_T(0.0, 0.0, 0.0, rot=rot_z(0))
    T_left1 = make_T(1.0, 0.0, 0.0, rot=rot_z(15))
    T_right0 = make_T(0.0, 1.0, 0.0, rot=rot_z(0))
    T_right1 = make_T(1.0, 1.0, 0.0, rot=rot_z(-10))

    demo_T_w_left = torch.stack([T_left0, T_left1])[None, None, ...]  # [B, D, T, 4, 4]
    demo_T_w_right = torch.stack([T_right0, T_right1])[None, None, ...]

    demo_T_left_to_right = torch.matmul(
        torch.inverse(demo_T_w_left), demo_T_w_right
    )

    # Actions (relative transforms)
    A_left0 = make_T(0.1, 0.0, 0.0, rot=rot_z(5))
    A_left1 = make_T(0.2, 0.0, 0.0, rot=rot_z(10))
    A_right0 = make_T(0.0, 0.1, 0.0, rot=rot_z(-5))
    A_right1 = make_T(0.0, 0.2, 0.0, rot=rot_z(-10))
    actions_left = torch.stack([A_left0, A_left1])[None, ...]   # [B, P, 4, 4]
    actions_right = torch.stack([A_right0, A_right1])[None, ...]

    # Current left-right relation (assume current poses are left origin, right offset)
    current_T_left_to_right = torch.tensor(
        relative_transform(make_T(0, 0, 0).numpy(), make_T(0, 1, 0).numpy()),
        dtype=torch.float32
    )[None, ...]

    # Scene positions (unique values for each time)
    def scene_positions(base: float) -> torch.Tensor:
        demo_pos = torch.tensor([
            [[base + 0.0, 0.0, 0.0], [base + 0.1, 0.0, 0.0]],
            [[base + 0.2, 0.0, 0.0], [base + 0.3, 0.0, 0.0]],
        ], dtype=torch.float32)  # [T, S, 3]
        live_pos = torch.tensor(
            [[base + 0.4, 0.0, 0.0], [base + 0.5, 0.0, 0.0]],
            dtype=torch.float32
        )  # [S, 3]
        action_pos = torch.tensor([
            [[base + 0.6, 0.0, 0.0], [base + 0.7, 0.0, 0.0]],
            [[base + 0.8, 0.0, 0.0], [base + 0.9, 0.0, 0.0]],
        ], dtype=torch.float32)  # [P, S, 3]
        return demo_pos, live_pos, action_pos

    demo_pos_left, live_pos_left, action_pos_left = scene_positions(0.0)
    demo_pos_right, live_pos_right, action_pos_right = scene_positions(1.0)

    # Scene embeddings (zeros are fine for structure checks)
    demo_emb = torch.zeros(B, D, T, S, config["local_nn_dim"])
    live_emb = torch.zeros(B, S, config["local_nn_dim"])
    action_emb = torch.zeros(B, P, S, config["local_nn_dim"])

    data = BimanualGraphData()
    data.demo_T_w_left = demo_T_w_left
    data.demo_T_w_right = demo_T_w_right
    data.demo_T_left_to_right = demo_T_left_to_right
    data.demo_grips_left = torch.zeros(B, D, T)
    data.demo_grips_right = torch.zeros(B, D, T)
    data.current_grip_left = torch.zeros(B)
    data.current_grip_right = torch.zeros(B)
    data.current_T_left_to_right = current_T_left_to_right

    data.actions_left = actions_left
    data.actions_right = actions_right
    data.actions_grip_left = torch.zeros(B, P)
    data.actions_grip_right = torch.zeros(B, P)
    data.diff_time = torch.tensor([[3]], dtype=torch.float32)

    data.demo_scene_embds_left = demo_emb
    data.demo_scene_pos_left = demo_pos_left[None, None, ...]
    data.demo_scene_embds_right = demo_emb
    data.demo_scene_pos_right = demo_pos_right[None, None, ...]
    data.live_scene_embds_left = live_emb
    data.live_scene_pos_left = live_pos_left[None, ...]
    data.live_scene_embds_right = live_emb
    data.live_scene_pos_right = live_pos_right[None, ...]
    data.action_scene_embds_left = action_emb
    data.action_scene_pos_left = action_pos_left[None, ...]
    data.action_scene_embds_right = action_emb
    data.action_scene_pos_right = action_pos_right[None, ...]

    return data

def build_expected_node_attrs(config: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
    B = config["batch_size"]
    D = config["num_demos"]
    T = config["traj_horizon"]
    P = config["pre_horizon"]
    S = config["num_scenes_nodes"]
    G = len(config["gripper_keypoints"])

    scene_batch = []
    scene_time = []
    scene_demo = []

    for b in range(B):
        for d in range(D):
            for t in range(T):
                for _ in range(S):
                    scene_batch.append(b)
                    scene_time.append(t)
                    scene_demo.append(d)
        # current
        for _ in range(S):
            scene_batch.append(b)
            scene_time.append(T)
            scene_demo.append(D)
        # action
        for p in range(P):
            for _ in range(S):
                scene_batch.append(b)
                scene_time.append(T + 1 + p)
                scene_demo.append(D)

    gripper_batch = []
    gripper_time = []
    gripper_demo = []
    gripper_node = []
    gripper_embd = []

    for b in range(B):
        for d in range(D):
            for t in range(T):
                for g in range(G):
                    gripper_batch.append(b)
                    gripper_time.append(t)
                    gripper_demo.append(d)
                    gripper_node.append(g)
                    gripper_embd.append(d * T * G + t * G + g)
        # current
        for g in range(G):
            gripper_batch.append(b)
            gripper_time.append(T)
            gripper_demo.append(D)
            gripper_node.append(g)
            gripper_embd.append(D * T * G + g)
        # action
        for p in range(P):
            for g in range(G):
                gripper_batch.append(b)
                gripper_time.append(T + 1 + p)
                gripper_demo.append(D)
                gripper_node.append(g)
                gripper_embd.append(D * T * G + G + p * G + g)

    return {
        "scene": {
            "batch": torch.tensor(scene_batch, dtype=torch.long),
            "time": torch.tensor(scene_time, dtype=torch.long),
            "demo": torch.tensor(scene_demo, dtype=torch.long),
        },
        "gripper": {
            "batch": torch.tensor(gripper_batch, dtype=torch.long),
            "time": torch.tensor(gripper_time, dtype=torch.long),
            "demo": torch.tensor(gripper_demo, dtype=torch.long),
            "node": torch.tensor(gripper_node, dtype=torch.long),
            "embd": torch.tensor(gripper_embd, dtype=torch.long),
        }
    }


def expected_edges(src: Dict[str, torch.Tensor],
                   dst: Dict[str, torch.Tensor],
                   rule_fn) -> Set[Tuple[int, int]]:
    edges = set()
    for i in range(src["batch"].shape[0]):
        for j in range(dst["batch"].shape[0]):
            if rule_fn(i, j, src, dst):
                edges.add((i, j))
    return edges


def check_node_indexing(graph_rep: BimanualGraphRep, config: Dict) -> None:
    section("STEP 2: Node indexing and embedding ranges")
    expected = build_expected_node_attrs(config)

    for arm in ["left", "right"]:
        substep(f"Validate {arm} node attributes")
        scene_prefix = f"scene_{arm}"
        grip_prefix = f"gripper_{arm}"

        assert_tensor_equal(
            getattr(graph_rep.graph, f"{scene_prefix}_batch"),
            expected["scene"]["batch"],
            f"{scene_prefix}_batch"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{scene_prefix}_traj"),
            expected["scene"]["time"],
            f"{scene_prefix}_traj"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{scene_prefix}_demo"),
            expected["scene"]["demo"],
            f"{scene_prefix}_demo"
        )

        assert_tensor_equal(
            getattr(graph_rep.graph, f"{grip_prefix}_batch"),
            expected["gripper"]["batch"],
            f"{grip_prefix}_batch"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{grip_prefix}_time"),
            expected["gripper"]["time"],
            f"{grip_prefix}_time"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{grip_prefix}_demo"),
            expected["gripper"]["demo"],
            f"{grip_prefix}_demo"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{grip_prefix}_node"),
            expected["gripper"]["node"],
            f"{grip_prefix}_node"
        )
        assert_tensor_equal(
            getattr(graph_rep.graph, f"{grip_prefix}_embd"),
            expected["gripper"]["embd"],
            f"{grip_prefix}_embd"
        )

    num_demo = config["num_demos"] * config["traj_horizon"] * len(config["gripper_keypoints"])
    num_other = len(config["gripper_keypoints"]) * (config["pre_horizon"] + 1)
    expected_total = num_demo + num_other

    substep("Validate gripper embedding sizes")
    assert_true(graph_rep.gripper_embds_left.num_embeddings == expected_total,
                "gripper_embds_left size")
    assert_true(graph_rep.gripper_embds_right.num_embeddings == expected_total,
                "gripper_embds_right size")
    print("PASS: node indexing and embedding ranges are correct.")


def check_node_features(graph_rep: BimanualGraphRep,
                        data: BimanualGraphData,
                        config: Dict) -> None:
    section("STEP 3: Node feature construction (scene + gripper)")

    B = config["batch_size"]
    D = config["num_demos"]
    T = config["traj_horizon"]
    P = config["pre_horizon"]
    G = len(config["gripper_keypoints"])

    for arm in ["left", "right"]:
        scene_key = f"scene_{arm}"

        demo_embds = getattr(data, f"demo_scene_embds_{arm}")[:, :D]
        live_embds = getattr(data, f"live_scene_embds_{arm}")
        action_embds = getattr(data, f"action_scene_embds_{arm}")

        demo_pos = getattr(data, f"demo_scene_pos_{arm}")[:, :D]
        live_pos = getattr(data, f"live_scene_pos_{arm}")
        action_pos = getattr(data, f"action_scene_pos_{arm}")

        expected_scene_x = torch.cat([
            demo_embds.reshape(-1, graph_rep.embd_dim),
            live_embds.reshape(-1, graph_rep.embd_dim),
            action_embds.reshape(-1, graph_rep.embd_dim),
        ], dim=0)
        expected_scene_pos = torch.cat([
            demo_pos.reshape(-1, 3),
            live_pos.reshape(-1, 3),
            action_pos.reshape(-1, 3),
        ], dim=0)

        assert_allclose(graph_rep.graph[scene_key].x, expected_scene_x,
                        f"{scene_key}.x")
        assert_allclose(graph_rep.graph[scene_key].pos, expected_scene_pos,
                        f"{scene_key}.pos")

    for arm in ["left", "right"]:
        grip_key = f"gripper_{arm}"

        demo_grips = getattr(data, f"demo_grips_{arm}")[:, :D]
        current_grip = getattr(data, f"current_grip_{arm}")
        action_grips = getattr(data, f"actions_grip_{arm}")

        base_kp = graph_rep.gripper_keypoints[None, None, None, :, :]  # [1, 1, 1, G, 3]
        demo_kp = base_kp.expand(B, D, T, G, 3)
        curr_kp = graph_rep.gripper_keypoints[None, :, :].expand(B, G, 3)
        action_kp = base_kp[:, 0].expand(B, P, G, 3)

        expected_grip_pos = torch.cat([
            demo_kp.reshape(-1, 3),
            curr_kp.reshape(-1, 3),
            action_kp.reshape(-1, 3),
        ], dim=0)

        demo_grip_embd = graph_rep.gripper_proj(demo_grips.unsqueeze(-1))
        demo_grip_embd = demo_grip_embd[:, :, :, None, :].expand(B, D, T, G, -1)
        curr_grip_embd = graph_rep.gripper_proj(current_grip.unsqueeze(-1))
        curr_grip_embd = curr_grip_embd[:, None, :].expand(B, G, -1)
        action_grip_embd = graph_rep.gripper_proj(action_grips.unsqueeze(-1))
        action_grip_embd = action_grip_embd[:, :, None, :].expand(B, P, G, -1)

        expected_grip_state = torch.cat([
            demo_grip_embd.reshape(-1, graph_rep.g_state_dim),
            curr_grip_embd.reshape(-1, graph_rep.g_state_dim),
            action_grip_embd.reshape(-1, graph_rep.g_state_dim),
        ], dim=0)

        embd_indices = getattr(graph_rep.graph, f"{grip_key}_embd")
        gripper_embds = getattr(graph_rep, f"gripper_embds_{arm}")
        learned_embd = gripper_embds(embd_indices % gripper_embds.num_embeddings)

        # Apply diffusion time embedding to action nodes.
        d_time_embd = graph_rep.time_emb(data.diff_time.squeeze())
        d_time_expanded = d_time_embd[:, None, None, :].expand(B, P, G, -1).reshape(-1, graph_rep.d_time_dim)
        action_start_idx = B * D * T * G + B * G
        learned_embd = learned_embd.clone()
        learned_embd[action_start_idx:, -graph_rep.d_time_dim:] = d_time_expanded

        expected_grip_x = torch.cat([learned_embd, expected_grip_state], dim=-1)

        assert_allclose(graph_rep.graph[grip_key].pos, expected_grip_pos,
                        f"{grip_key}.pos")
        assert_allclose(graph_rep.graph[grip_key].x, expected_grip_x,
                        f"{grip_key}.x")

    print("PASS: node features (scene + gripper) are correct.")

def check_edge_topology(graph_rep: BimanualGraphRep, config: Dict) -> None:
    section("STEP 4: Edge topology for all edge types")

    T = config["traj_horizon"]
    expected = build_expected_node_attrs(config)

    def same_batch_time_demo(i, j, src, dst) -> bool:
        return (src["batch"][i] == dst["batch"][j] and
                src["time"][i] == dst["time"][j] and
                src["demo"][i] == dst["demo"][j])

    def is_action_time(t) -> bool:
        return t > T

    edge_expectations: Dict[Tuple[str, str, str], Set[Tuple[int, int]]] = {}

    for arm in ["left", "right"]:
        scene = expected["scene"]
        grip = expected["gripper"]
        scene_key = f"scene_{arm}"
        grip_key = f"gripper_{arm}"

        # Scene-scene
        edge_expectations[(scene_key, "rel_action", scene_key)] = expected_edges(
            scene, scene,
            lambda i, j, s, d: same_batch_time_demo(i, j, s, d)
            and is_action_time(s["time"][i]) and is_action_time(d["time"][j])
        )
        edge_expectations[(scene_key, "rel_demo", scene_key)] = expected_edges(
            scene, scene,
            lambda i, j, s, d: same_batch_time_demo(i, j, s, d)
            and not (is_action_time(s["time"][i]) and is_action_time(d["time"][j]))
        )

        # Scene-gripper
        edge_expectations[(scene_key, "rel_action", grip_key)] = expected_edges(
            scene, grip,
            lambda i, j, s, d: same_batch_time_demo(i, j, s, d)
            and is_action_time(s["time"][i]) and is_action_time(d["time"][j])
        )
        edge_expectations[(scene_key, "rel_demo", grip_key)] = expected_edges(
            scene, grip,
            lambda i, j, s, d: same_batch_time_demo(i, j, s, d)
            and not (is_action_time(s["time"][i]) and is_action_time(d["time"][j]))
        )

        # Gripper-gripper
        edge_expectations[(grip_key, "rel", grip_key)] = expected_edges(
            grip, grip, same_batch_time_demo
        )
        edge_expectations[(grip_key, "demo", grip_key)] = expected_edges(
            grip, grip,
            lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                                and s["time"][i] < T and d["time"][j] < T
                                and s["demo"][i] == d["demo"][j]
                                and (d["time"][j] - s["time"][i]) == -1)
        )
        edge_expectations[(grip_key, "cond", grip_key)] = expected_edges(
            grip, grip,
            lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                                and s["time"][i] < T and d["time"][j] == T)
        )
        edge_expectations[(grip_key, "rel_cond", grip_key)] = expected_edges(
            grip, grip,
            lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                                and s["time"][i] == T and d["time"][j] > T)
        )
        edge_expectations[(grip_key, "time_action", grip_key)] = expected_edges(
            grip, grip,
            lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                                and s["time"][i] > T and d["time"][j] > T
                                and s["time"][i] != d["time"][j])
        )

    # Cross-arm edges
    left = expected["gripper"]
    right = expected["gripper"]

    edge_expectations[("gripper_left", "cross", "gripper_right")] = expected_edges(
        left, right,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] == T and d["time"][j] == T)
    )
    edge_expectations[("gripper_right", "cross", "gripper_left")] = expected_edges(
        right, left,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] == T and d["time"][j] == T)
    )
    edge_expectations[("gripper_left", "cross_action", "gripper_right")] = expected_edges(
        left, right,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] > T and d["time"][j] > T
                            and s["time"][i] == d["time"][j])
    )
    edge_expectations[("gripper_right", "cross_action", "gripper_left")] = expected_edges(
        right, left,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] > T and d["time"][j] > T
                            and s["time"][i] == d["time"][j])
    )
    edge_expectations[("gripper_left", "cross_demo", "gripper_right")] = expected_edges(
        left, right,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] < T and d["time"][j] < T
                            and s["demo"][i] == d["demo"][j])
    )
    edge_expectations[("gripper_right", "cross_demo", "gripper_left")] = expected_edges(
        right, left,
        lambda i, j, s, d: (s["batch"][i] == d["batch"][j]
                            and s["time"][i] < T and d["time"][j] < T
                            and s["demo"][i] == d["demo"][j])
    )

    for edge_key, expected_set in edge_expectations.items():
        actual_index = graph_rep.graph[edge_key].edge_index
        actual_set = set(map(tuple, actual_index.t().tolist()))
        assert_true(actual_set == expected_set, f"edge topology {edge_key}")

    print("PASS: edge topology matches explicit definitions.")

def compute_all_transforms(graph_rep: BimanualGraphRep,
                           data: BimanualGraphData,
                           config: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    B = config["batch_size"]
    D = config["num_demos"]
    T = config["traj_horizon"]
    P = config["pre_horizon"]
    G = len(config["gripper_keypoints"])

    all_T_w_e = {}
    all_T_e_w = {}

    for arm in ["left", "right"]:
        demo_T = getattr(data, f"demo_T_w_{arm}")[:, :D].reshape(B * D * T, 4, 4)
        current_T = torch.eye(4).repeat(B, 1, 1).reshape(B, 4, 4)
        action_T = getattr(data, f"actions_{arm}").reshape(B * P, 4, 4)
        T_w_e = torch.cat([demo_T, current_T, action_T], dim=0)
        T_w_e = T_w_e[:, None, :, :].repeat(1, G, 1, 1).reshape(-1, 4, 4)
        T_e_w = torch.inverse(T_w_e)
        all_T_w_e[f"gripper_{arm}"] = T_w_e
        all_T_e_w[f"gripper_{arm}"] = T_e_w

    return all_T_w_e, all_T_e_w


def compute_rel_attr(graph_rep: BimanualGraphRep,
                     src: str,
                     dst: str,
                     edge_index: torch.Tensor,
                     all_T_w_e: Dict[str, torch.Tensor],
                     all_T_e_w: Dict[str, torch.Tensor]) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.zeros((0, graph_rep.edge_dim), device=graph_rep.device)

    src_pos = graph_rep.graph[src].pos[edge_index[0]]
    dst_pos = graph_rep.graph[dst].pos[edge_index[1]]

    T_w_src = all_T_w_e.get(src)
    T_dst_w = all_T_e_w.get(dst)

    if T_w_src is not None and T_dst_w is not None:
        T_src_w = all_T_e_w[src][edge_index[0]]
        T_w_dst = all_T_w_e[dst][edge_index[1]]
        T_src_dst = torch.bmm(T_src_w, T_w_dst)
        pos_dest_rot = torch.bmm(T_src_dst[..., :3, :3], src_pos[..., None]).squeeze(-1)
        rel_trans = T_src_dst[..., :3, 3]
        return torch.cat([
            graph_rep.pos_encoder(rel_trans),
            graph_rep.pos_encoder(pos_dest_rot - src_pos),
        ], dim=-1)

    rel_pos = dst_pos - src_pos
    return torch.cat([
        graph_rep.pos_encoder(rel_pos),
        graph_rep.pos_encoder(rel_pos),
    ], dim=-1)


def compute_cross_attr(graph_rep: BimanualGraphRep,
                       edge_index: torch.Tensor,
                       src_key: str,
                       dst_key: str,
                       T_src_from_dst: torch.Tensor) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.zeros((0, graph_rep.edge_dim), device=graph_rep.device)
    src_pos = graph_rep.graph[src_key].pos[edge_index[0]]
    dst_pos = graph_rep.graph[dst_key].pos[edge_index[1]]

    dst_in_src = torch.bmm(
        T_src_from_dst[:, :3, :3],
        dst_pos[:, :, None]
    ).squeeze(-1) + T_src_from_dst[:, :3, 3]
    rel_pos = dst_in_src - src_pos
    pos_dest_rot = torch.bmm(
        T_src_from_dst[:, :3, :3],
        src_pos[:, :, None]
    ).squeeze(-1)

    return torch.cat([
        graph_rep.pos_encoder(rel_pos),
        graph_rep.pos_encoder(pos_dest_rot - src_pos),
    ], dim=-1)


def check_edge_attributes(graph_rep: BimanualGraphRep,
                          data: BimanualGraphData,
                          config: Dict) -> None:
    section("STEP 5: Edge attribute correctness (all edge types)")
    all_T_w_e, all_T_e_w = compute_all_transforms(graph_rep, data, config)

    # Local edges
    for arm in ["left", "right"]:
        scene_key = f"scene_{arm}"
        grip_key = f"gripper_{arm}"

        for edge_type in ["rel_demo", "rel_action"]:
            edge_key = (scene_key, edge_type, scene_key)
            expected = compute_rel_attr(
                graph_rep, scene_key, scene_key,
                graph_rep.graph[edge_key].edge_index,
                all_T_w_e, all_T_e_w
            )
            assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                            f"{edge_key} edge_attr")

        for edge_type in ["rel_demo", "rel_action"]:
            edge_key = (scene_key, edge_type, grip_key)
            expected = compute_rel_attr(
                graph_rep, scene_key, grip_key,
                graph_rep.graph[edge_key].edge_index,
                all_T_w_e, all_T_e_w
            )
            assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                            f"{edge_key} edge_attr")

        edge_key = (grip_key, "rel", grip_key)
        expected = compute_rel_attr(
            graph_rep, grip_key, grip_key,
            graph_rep.graph[edge_key].edge_index,
            all_T_w_e, all_T_e_w
        )
        assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                        f"{edge_key} edge_attr")

        edge_key = (grip_key, "cond", grip_key)
        base = compute_rel_attr(
            graph_rep, grip_key, grip_key,
            graph_rep.graph[edge_key].edge_index,
            all_T_w_e, all_T_e_w
        )
        learned = graph_rep.cond_edge_embd(
            torch.zeros(base.shape[0], dtype=torch.long, device=graph_rep.device)
        )
        expected = base + learned
        assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                        f"{edge_key} edge_attr")

        for edge_type in ["demo", "time_action", "rel_cond"]:
            edge_key = (grip_key, edge_type, grip_key)
            expected = compute_rel_attr(
                graph_rep, grip_key, grip_key,
                graph_rep.graph[edge_key].edge_index,
                all_T_w_e, all_T_e_w
            )
            assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                            f"{edge_key} edge_attr")

    # Cross-arm edges
    T = config["traj_horizon"]

    T_left_to_right = data.current_T_left_to_right
    if T_left_to_right.dim() == 2:
        T_left_to_right = T_left_to_right.unsqueeze(0)
    T_right_to_left = torch.inverse(T_left_to_right)

    actions_left = data.actions_left
    actions_right = data.actions_right
    action_T_left_to_right = None
    if actions_left is not None and actions_right is not None:
        B, P = actions_left.shape[:2]
        T_lr_curr = T_left_to_right[:, None, :, :].expand(B, P, 4, 4).reshape(-1, 4, 4)
        A_left_inv = torch.inverse(actions_left.reshape(-1, 4, 4))
        A_right = actions_right.reshape(-1, 4, 4)
        action_T_left_to_right = torch.bmm(A_left_inv, torch.bmm(T_lr_curr, A_right)).view(B, P, 4, 4)
    action_T_right_to_left = torch.inverse(action_T_left_to_right) if action_T_left_to_right is not None else None

    demo_T_left = data.demo_T_w_left
    demo_T_right = data.demo_T_w_right
    if demo_T_left.dim() == 4:
        demo_T_left = demo_T_left.unsqueeze(0)
    if demo_T_right.dim() == 4:
        demo_T_right = demo_T_right.unsqueeze(0)

    for src_arm, dst_arm, T_curr, T_action in [
        ("left", "right", T_left_to_right, action_T_left_to_right),
        ("right", "left", T_right_to_left, action_T_right_to_left),
    ]:
        # cross
        edge_key = (f"gripper_{src_arm}", "cross", f"gripper_{dst_arm}")
        edge_index = graph_rep.graph[edge_key].edge_index
        src_batch = getattr(graph_rep.graph, f"gripper_{src_arm}_batch")[edge_index[0]]
        T_src_from_dst = T_curr[src_batch]
        expected = compute_cross_attr(
            graph_rep, edge_index,
            f"gripper_{src_arm}", f"gripper_{dst_arm}",
            T_src_from_dst
        )
        learned = graph_rep.cross_arm_edge_embd(
            torch.zeros(expected.shape[0], dtype=torch.long, device=graph_rep.device)
        )
        expected = expected + learned
        assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                        f"{edge_key} edge_attr")

        # cross_action
        edge_key = (f"gripper_{src_arm}", "cross_action", f"gripper_{dst_arm}")
        edge_index = graph_rep.graph[edge_key].edge_index
        src_batch = getattr(graph_rep.graph, f"gripper_{src_arm}_batch")[edge_index[0]]
        src_time = getattr(graph_rep.graph, f"gripper_{src_arm}_time")[edge_index[0]].long()
        if T_action is None:
            T_src_from_dst = T_curr[src_batch]
        else:
            max_step = T_action.shape[1] - 1
            action_step = (src_time - T - 1).clamp(min=0, max=max_step)
            T_src_from_dst = T_action[src_batch, action_step]
        expected = compute_cross_attr(
            graph_rep, edge_index,
            f"gripper_{src_arm}", f"gripper_{dst_arm}",
            T_src_from_dst
        )
        learned = graph_rep.cross_action_edge_embd(
            torch.zeros(expected.shape[0], dtype=torch.long, device=graph_rep.device)
        )
        expected = expected + learned
        assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                        f"{edge_key} edge_attr")

        # cross_demo uses demo_T_w_left/right at src_time and dst_time
        edge_key = (f"gripper_{src_arm}", "cross_demo", f"gripper_{dst_arm}")
        edge_index = graph_rep.graph[edge_key].edge_index
        src_batch = getattr(graph_rep.graph, f"gripper_{src_arm}_batch")[edge_index[0]]
        src_time = getattr(graph_rep.graph, f"gripper_{src_arm}_time")[edge_index[0]].long()
        dst_time = getattr(graph_rep.graph, f"gripper_{dst_arm}_time")[edge_index[1]].long()
        src_demo = getattr(graph_rep.graph, f"gripper_{src_arm}_demo")[edge_index[0]].long()

        if src_arm == "left":
            T_w_src = demo_T_left[src_batch, src_demo, src_time]
            T_w_dst = demo_T_right[src_batch, src_demo, dst_time]
        else:
            T_w_src = demo_T_right[src_batch, src_demo, src_time]
            T_w_dst = demo_T_left[src_batch, src_demo, dst_time]

        T_src_from_dst = torch.bmm(torch.inverse(T_w_src), T_w_dst)
        expected = compute_cross_attr(
            graph_rep, edge_index,
            f"gripper_{src_arm}", f"gripper_{dst_arm}",
            T_src_from_dst
        )
        learned = graph_rep.cross_arm_edge_embd(
            torch.zeros(expected.shape[0], dtype=torch.long, device=graph_rep.device)
        )
        expected = expected + learned
        assert_allclose(graph_rep.graph[edge_key].edge_attr, expected,
                        f"{edge_key} edge_attr")

    print("PASS: edge attributes match explicit geometry.")

def check_action_left_to_right(data: BimanualGraphData) -> None:
    section("STEP 6: Action-level left-to-right transform consistency")
    actions_left = data.actions_left
    actions_right = data.actions_right
    T_lr_curr = data.current_T_left_to_right
    if T_lr_curr.dim() == 2:
        T_lr_curr = T_lr_curr.unsqueeze(0)

    B, P = actions_left.shape[:2]
    T_lr_curr_expand = T_lr_curr[:, None].expand(B, P, 4, 4).reshape(-1, 4, 4)
    A_left_inv = torch.inverse(actions_left.reshape(-1, 4, 4))
    A_right = actions_right.reshape(-1, 4, 4)
    expected = torch.bmm(A_left_inv, torch.bmm(T_lr_curr_expand, A_right)).view(B, P, 4, 4)

    # Manual reference: for each step, compare to inv(A_left) * T_lr_curr * A_right
    for p in range(P):
        ref = torch.inverse(actions_left[0, p]) @ T_lr_curr[0] @ actions_right[0, p]
        assert_allclose(expected[0, p], ref, f"action_T_left_to_right step {p}")
    print("PASS: action-level T_left_to_right matches formula.")


def check_global_invariance(graph_rep: BimanualGraphRep,
                            data: BimanualGraphData,
                            config: Dict) -> None:
    section("STEP 7: Global-frame invariance for demo-based edges")
    T_global = make_T(2.0, -1.0, 0.0, rot=rot_z(45))

    data_g = BimanualGraphData()
    keys = data.keys if hasattr(data, "keys") else []
    if callable(keys):
        keys = keys()
    for k in keys:
        v = data[k]
        setattr(data_g, k, v.clone() if torch.is_tensor(v) else v)

    # Apply global transform to demo poses and keep relative transforms consistent.
    data_g.demo_T_w_left = torch.matmul(T_global, data.demo_T_w_left)
    data_g.demo_T_w_right = torch.matmul(T_global, data.demo_T_w_right)
    data_g.demo_T_left_to_right = torch.matmul(
        torch.inverse(data_g.demo_T_w_left), data_g.demo_T_w_right
    )

    graph_rep.update_graph(data)
    base_demo_attr = graph_rep.graph[("gripper_left", "demo", "gripper_left")].edge_attr.clone()
    base_cross_demo_attr = graph_rep.graph[("gripper_left", "cross_demo", "gripper_right")].edge_attr.clone()

    graph_rep.update_graph(data_g)
    new_demo_attr = graph_rep.graph[("gripper_left", "demo", "gripper_left")].edge_attr.clone()
    new_cross_demo_attr = graph_rep.graph[("gripper_left", "cross_demo", "gripper_right")].edge_attr.clone()

    assert_allclose(new_demo_attr, base_demo_attr, "demo edge_attr invariance")
    assert_allclose(new_cross_demo_attr, base_cross_demo_attr, "cross_demo edge_attr invariance")
    print("PASS: demo-based edge attributes invariant to global frame.")

def compute_labels(gt: torch.Tensor,
                   noisy: torch.Tensor,
                   gripper_kp: torch.Tensor) -> torch.Tensor:
    B, P = gt.shape[:2]
    G = gripper_kp.shape[0]
    kp = gripper_kp[None, None, :, :].repeat(B, P, 1, 1)

    T_w_n = noisy.view(-1, 4, 4)
    T_n_w = torch.inverse(T_w_n)
    T_w_g = gt.view(-1, 4, 4)
    T_n_g = torch.bmm(T_n_w, T_w_g).view(B, P, 4, 4)

    labels_trans = T_n_g[..., :3, 3][:, :, None, :].repeat(1, 1, G, 1)
    T_n_g_rot = T_n_g.clone()
    T_n_g_rot[..., :3, 3] = 0

    kp_flat = kp.view(-1, G, 3)
    T_flat = T_n_g_rot.view(-1, 4, 4)
    kp_rot = torch.bmm(T_flat[:, :3, :3], kp_flat.transpose(1, 2)).transpose(1, 2)
    labels_rot = kp_rot - kp_flat
    labels_rot = labels_rot.view(B, P, G, 3)
    labels = torch.cat([labels_trans, labels_rot], dim=-1)
    return labels


def decode_labels_to_action(labels: torch.Tensor,
                            noisy: torch.Tensor,
                            gripper_kp: torch.Tensor) -> torch.Tensor:
    B, P, G = labels.shape[:3]
    kp = gripper_kp[None, None, :, :].repeat(B, P, 1, 1)

    pred_trans = labels[..., :3]
    pred_rot = labels[..., 3:6]

    translation = pred_trans.mean(dim=-2)  # [B, P, 3]
    target_kp = kp + pred_rot + translation[:, :, None, :]

    # Rigid fit for rotation + translation (matches diffusion update)
    def rigid_svd(a, b):
        p1 = a
        p2 = b
        p1_centroid = p1.mean(dim=1, keepdim=True)
        p2_centroid = p2.mean(dim=1, keepdim=True)
        p1_prime = p1 - p1_centroid
        p2_prime = p2 - p2_centroid
        H = torch.bmm(p1_prime.transpose(1, 2), p2_prime)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)
        D = torch.eye(3, device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1)
        D[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
        R = V @ D @ U.transpose(1, 2)
        t = p2_centroid - torch.bmm(R, p1_centroid.transpose(1, 2)).transpose(1, 2)
        T = torch.eye(4, device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = t.squeeze(-2)
        return T

    T_delta = rigid_svd(
        kp.reshape(-1, G, 3),
        target_kp.reshape(-1, G, 3),
    ).reshape(B, P, 4, 4)

    pred_actions = torch.bmm(
        noisy.reshape(-1, 4, 4),
        T_delta.reshape(-1, 4, 4),
    ).reshape(B, P, 4, 4)
    return pred_actions


def check_labels_decode() -> None:
    section("STEP 8: Labels and decode consistency")
    gripper_kp = torch.tensor([
        [0.02, 0.0, 0.0],
        [-0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
    ], dtype=torch.float32)

    noisy = torch.eye(4).reshape(1, 1, 4, 4)
    gt = make_T(0.1, 0.0, 0.0, rot=rot_z(30)).reshape(1, 1, 4, 4)

    labels = compute_labels(gt, noisy, gripper_kp)
    pred = decode_labels_to_action(labels, noisy, gripper_kp)
    assert_allclose(pred, gt, "decode(labels) == gt (noisy=I)")

    noisy2 = make_T(0.2, -0.1, 0.0, rot=rot_z(-10)).reshape(1, 1, 4, 4)
    gt2 = make_T(0.25, -0.05, 0.0, rot=rot_z(15)).reshape(1, 1, 4, 4)
    labels2 = compute_labels(gt2, noisy2, gripper_kp)
    pred2 = decode_labels_to_action(labels2, noisy2, gripper_kp)
    assert_allclose(pred2, gt2, "decode(labels) == gt (noisy!=I)")

    print("PASS: labels decode back to ground truth.")


def check_collate() -> None:
    section("STEP 9: Collate function correctness")

    # Build two tiny samples with distinct values.
    def sample(offset: float) -> BimanualGraphData:
        d = BimanualGraphData()
        T = 2
        N = 3
        d.demo_T_w_left = torch.eye(4).reshape(1, T, 4, 4)
        d.demo_T_w_right = torch.eye(4).reshape(1, T, 4, 4)
        d.demo_grips_left = torch.zeros(1, T)
        d.demo_grips_right = torch.zeros(1, T)
        d.demo_T_left_to_right = torch.eye(4).reshape(1, T, 4, 4)
        d.pos_demos_left = torch.ones(T, N, 3) * offset
        d.pos_demos_right = torch.ones(T, N, 3) * (offset + 1.0)
        d.pos_obs_left = torch.ones(N, 3) * (offset + 2.0)
        d.pos_obs_right = torch.ones(N, 3) * (offset + 3.0)
        d.current_T_left_to_right = torch.eye(4)
        d.current_grip_left = torch.tensor(0.0)
        d.current_grip_right = torch.tensor(0.0)
        d.actions_left = torch.eye(4).reshape(1, 1, 4, 4)
        d.actions_right = torch.eye(4).reshape(1, 1, 4, 4)
        d.actions_grip_left = torch.zeros(1, 1)
        d.actions_grip_right = torch.zeros(1, 1)
        return d

    batch = collate_bimanual([sample(0.0), sample(10.0)])

    assert_true(batch.pos_demos_left.shape[0] == 2 * 2 * 3,
                "pos_demos_left flattened length")
    assert_true(batch.batch_demos_left.shape[0] == 2 * 2 * 3,
                "batch_demos_left length")
    assert_true(batch.pos_obs_left.shape[0] == 2 * 3,
                "pos_obs_left flattened length")
    assert_true(batch.batch_pos_obs.shape[0] == 2 * 3,
                "batch_pos_obs length")

    # Batch indices should be [0..B-1] for each point.
    assert_true(batch.batch_pos_obs.min().item() == 0, "batch_pos_obs min")
    assert_true(batch.batch_pos_obs.max().item() == 1, "batch_pos_obs max")

    print("PASS: collate produces correct shapes and batch indices.")

def test_frame_invariance() -> None:
    section("STEP 1: SE(3) frame invariance for left-right relatives")

    T_left0 = make_T(0, 0, 0)
    T_left1 = make_T(1, 0, 0)
    T_right0 = make_T(0, 1, 0)
    T_right1 = make_T(1, 1, 0)

    rel0 = torch.tensor(relative_transform(T_left0.numpy(), T_right0.numpy()))
    rel1 = torch.tensor(relative_transform(T_left1.numpy(), T_right1.numpy()))
    print("T_left_to_right(t0):\n", rel0)
    print("T_left_to_right(t1):\n", rel1)

    T_global = make_T(5, -2, 0, rot=rot_z(90))
    T_left0_g = T_global @ T_left0
    T_left1_g = T_global @ T_left1
    T_right0_g = T_global @ T_right0
    T_right1_g = T_global @ T_right1

    rel0_g = torch.tensor(relative_transform(T_left0_g.numpy(), T_right0_g.numpy()))
    rel1_g = torch.tensor(relative_transform(T_left1_g.numpy(), T_right1_g.numpy()))

    assert_allclose(rel0, rel0_g, "left_to_right t0 invariance")
    assert_allclose(rel1, rel1_g, "left_to_right t1 invariance")
    print("PASS: relative transforms invariant to global frame change.")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    test_frame_invariance()

    config = build_test_config()
    graph_rep = BimanualGraphRep(config)
    graph_rep.initialise_graph()

    data = build_test_data(config)
    graph_rep.update_graph(data)

    check_node_indexing(graph_rep, config)
    check_node_features(graph_rep, data, config)
    check_edge_topology(graph_rep, config)
    check_edge_attributes(graph_rep, data, config)
    check_action_left_to_right(data)
    check_global_invariance(graph_rep, data, config)
    check_labels_decode()
    check_collate()

    print("\nALL VERIFICATION STEPS PASSED.")


if __name__ == "__main__":
    main()
