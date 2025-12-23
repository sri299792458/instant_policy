# Bimanual Instant Policy: The Complete Technical Reference

**A Deep Technical Document for Readers Familiar with Single-Arm Instant Policy**

This document provides an exhaustive technical breakdown of how single-arm Instant Policy is extended to bimanual manipulation. Every architectural decision, mathematical formulation, and implementation detail is explained with the goal of enabling complete understanding without needing to read the source code.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Bimanual Challenge: Why It's Hard](#2-the-bimanual-challenge-why-its-hard)
3. [Architecture Overview](#3-architecture-overview)
4. [Graph Representation Deep Dive](#4-graph-representation-deep-dive)
5. [Model Architecture In Detail](#5-model-architecture-in-detail)
6. [Diffusion Process](#6-diffusion-process)
7. [Pseudo-Demonstration Generation](#7-pseudo-demonstration-generation)
8. [Dataset & Data Pipeline](#8-dataset--data-pipeline)
9. [Training Details](#9-training-details)
10. [Configuration Reference](#10-configuration-reference)
11. [File-by-File Reference](#11-file-by-file-reference)

---

## 1. Executive Summary

### What This Codebase Does

This codebase extends Instant Policy (IP) from single-arm to bimanual manipulation. The core insight is that bimanual manipulation requires:

1. **Dual egocentric representation** - Each arm sees the world from its own frame
2. **Cross-arm coordination** - Arms must know about each other's state and intentions
3. **Selective gating** - Not all tasks need tight coordination (independent vs. synchronized)
4. **Coupled diffusion** - Both arms denoise together for coherent predictions

### Quick Comparison: Single-Arm vs. Bimanual

| Aspect         | Single-Arm IP      | Bimanual Extension                                           |
| -------------- | ------------------ | ------------------------------------------------------------ |
| Node Types     | `scene`, `gripper` | `scene_left`, `scene_right`, `gripper_left`, `gripper_right` |
| Edge Types     | 7 types            | 18+ types (includes cross-arm edges)                         |
| Scene Encoding | 1 egocentric frame | 2 egocentric frames (left EE, right EE)                      |
| Outputs        | `[B, P, G, 7]`     | Two outputs: `preds_left`, `preds_right`                     |
| Diffusion      | Single process     | Coupled process (same timestep, coordinated)                 |
| Coordination   | N/A                | Learnable gating + optional consistency loss                 |

### File Overview

```
bimanual_instant_policy/
├── src/                          # Main source code
│   ├── models/
│   │   ├── graph_rep.py          # 989 lines - Bimanual graph structure, edges, attributes
│   │   ├── model.py              # 483 lines - BimanualAGI model
│   │   └── diffusion.py          # 515 lines - Coupled diffusion training
│   ├── data/
│   │   ├── dataset.py            # 434 lines - Dataset classes
│   │   ├── generator.py          # 605 lines - Pseudo-demo generation
│   │   ├── data_structures.py    # 241 lines - Core data types
│   │   ├── objects.py            # 463 lines - Procedural object generation
│   │   ├── prepare_data.py       # Data preparation utilities
│   │   └── patterns/
│   │       └── coordination_patterns.py  # 443 lines - Pattern generators
│   └── evaluation/
│       ├── eval.py               # Main evaluation script
│       ├── rl_bench_utils.py     # RLBench utilities
│       ├── rl_bench_tasks.py     # Task definitions
│       └── deployment.py         # Deployment utilities
├── scripts/
│   ├── train.py                  # 212 lines - Training script
│   ├── profile_training.py       # Profiling utilities
│   ├── verify_extension.py       # Extension verification
│   └── slurm_bimanual_train.sh   # SLURM training job
├── external/                     # External dependencies
│   └── ip/                       # Original single-arm Instant Policy
│       ├── models/
│       │   ├── graph_rep.py      # Original graph representation
│       │   ├── model.py          # Original AGI model
│       │   └── scene_encoder.py  # PointNet++ encoder (shared)
│       └── configs/
│           └── bimanual_config.py  # Configuration
├── apptainer/                    # Container setup for HPC
│   ├── rlbench.def               # Apptainer definition
│   ├── build_container.sh        # Build script
│   └── run_rlbench_vnc.sh        # Run script with VNC
└── docs/                         # Documentation
```

---

## 2. The Bimanual Challenge: Why It's Hard

### The Fundamental Problem

Single-arm IP works because everything is transformed to the end-effector frame, making the policy **spatially invariant**. But with two arms:

- Which frame do we use?
- How do we represent the relationship between arms?
- How do we NOT break invariances that make the single-arm approach work?

### Failed Approaches (What NOT to Do)

**❌ Single Global Frame (Robot Base)**
```
World Frame
├── Left Arm Pose (relative to world)
├── Right Arm Pose (relative to world)
└── Scene Points (in world frame)
```
**Problem**: Policy learns poses relative to the base. Moving the robot invalidates the policy.

**❌ Single Egocentric Frame (e.g., Left EE)**
```
Left EE Frame
├── Left Arm: Always at origin
├── Right Arm Pose (relative to left)
└── Scene Points (in left frame)
```
**Problem**: Asymmetric representation. Left and right arms learn fundamentally different policies. Poor generalization.

### The Correct Approach: Dual Egocentric Representation

```
Left EE Frame                    Right EE Frame
├── Left Arm: At origin          ├── Right Arm: At origin
├── Scene in Left Frame          ├── Scene in Right Frame
└── Right visible via EDGE       └── Left visible via EDGE
                    ↘             ↙
                 Cross-Arm Edges
              (Relative Transforms)
```

**Why This Works:**
1. **Translation invariance** - Moving both arms together doesn't change learning
2. **Rotation invariance** - Rotating the entire scene works correctly
3. **Symmetry** - Both arms get equivalent representation
4. **Base-motion invariance** - Robot base pose doesn't matter

The key insight: **Cross-arm relationships are encoded as edges with relative SE(3) transforms as attributes**, not as node features.

---

## 3. Architecture Overview

### High-Level Flow

```
Input: BimanualGraphData
├── Point clouds (left/right frames)
├── Demo trajectories (both arms, T waypoints)
├── Current state (both arms)
└── Noisy actions (both arms, P steps)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Scene Encoding (Per Arm)                             │
│                                                                             │
│  demo_pointcloud_left_frame  → SceneEncoder → demo_scene_embeddings_left    │
│  demo_pointcloud_right_frame → SceneEncoder → demo_scene_embeddings_right   │
│  current_pointcloud_left_frame  → SceneEncoder → live_scene_embeddings_left │
│  current_pointcloud_right_frame → SceneEncoder → live_scene_embeddings_right│
│                                                                             │
│  (All point clouds are in the respective arm's egocentric coordinate frame)│
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│    Graph Construction               │
│  BimanualGraphRep.update_graph()    │
│  - Node features: scene + gripper   │
│  - Edge indices: 18 edge types      │
│  - Edge attributes: relative geom   │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Three-Stage Graph Transformer      │
│                                     │
│  1. LOCAL ENCODER                   │
│     - Process rel edges (spatial)   │
│     - NO cross-arm edges yet        │
│                                     │
│  2. COND ENCODER                    │
│     - Process demo, cond edges      │
│     - Process cross, cross_demo     │
│     - (After gating applied)        │
│                                     │
│  3. ACTION ENCODER                  │
│     - Process time_action, rel_cond │
│     - Process cross_action          │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│   Prediction Heads (Per Arm)        │
│                                     │
│  left_gripper_features[action_mask] │
│    → MLPs → translation_delta_left, │
│             rotation_delta_left,    │
│             gripper_state_left      │
│                                     │
│  right_gripper_features[action_mask]│
│    → MLPs → translation_delta_right,│
│             rotation_delta_right,   │
│             gripper_state_right     │
└────────────────────────────────────┘
         │
         ▼
Output: (predictions_left, predictions_right)
        [batch_size, prediction_horizon, gripper_keypoints, 7] each
```

---

## 4. Graph Representation Deep Dive

**File**: `src/models/graph_rep.py` (989 lines)

### 4.1 Node Types

The bimanual graph has **4 node types** (vs. 2 in single-arm):

| Node Type       | What It Represents             | Count per Sample    |
| --------------- | ------------------------------ | ------------------- |
| `scene_left`    | Scene points in left EE frame  | B×(D×T×S + S + P×S) |
| `scene_right`   | Scene points in right EE frame | B×(D×T×S + S + P×S) |
| `gripper_left`  | Left gripper keypoints         | B×(D×T×G + G + P×G) |
| `gripper_right` | Right gripper keypoints        | B×(D×T×G + G + P×G) |

Where:
- batch_size = number of samples per training batch (default: 8)
- num_demos = number of demonstrations provided as context (default: 1)
- trajectory_horizon = waypoints per demo trajectory (default: 10)
- scene_nodes = PointNet++ output nodes per arm (default: 16)
- gripper_keypoints = keypoints defining gripper geometry (6 points)
- prediction_horizon = future action steps to predict (default: 8)

**Example calculation** (D=1, T=10, P=8, S=16, G=6):
- Scene nodes per arm: 1×10×16 + 16 + 8×16 = 304
- Gripper nodes per arm: 1×10×6 + 6 + 8×6 = 114
- **Total per arm: 418 nodes**
- **Total both arms: 836 nodes**

### 4.2 The Three Node Categories

For each arm, nodes are divided into three categories based on temporal role:

```
Demo Nodes (Past)              Current Nodes         Action Nodes (Future)
──────────────────────────┬───────────────────┬─────────────────────────────
time: [0, traj_horizon-1]  │ time: traj_horizon │ time: [traj_horizon+1, ...]
                           │                    │
Scene: [batch, demos,      │ Scene: [batch,     │ Scene: [batch, pred_horizon,
        traj_horizon,      │        scene_nodes,│         scene_nodes, 3]
        scene_nodes, 3]    │        3]          │
Gripper: [batch, demos,    │ Gripper: [batch,   │ Gripper: [batch, pred_horizon,
          traj_horizon,    │          keypoints,│           keypoints, 3]
          keypoints, 3]    │          3]        │
```

This temporal structure is crucial for understanding edge connectivity.

### 4.3 Graph Structure Visualization

**1. TEMPORAL EDGES (within demo sequence)**
```
    Demo t=0         Demo t=1         Demo t=2    ...    Demo t=T-1
   ┌─────────┐      ┌─────────┐      ┌─────────┐        ┌─────────┐
   │ Gripper │ ◄─── │ Gripper │ ◄─── │ Gripper │  ...   │ Gripper │
   └─────────┘      └─────────┘      └─────────┘        └─────────┘
                 temporal              temporal
```

**2. CONTEXT EDGES (demo → current)**
```
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Demo    │ │ Demo    │ │ Demo    │
   │ t=0     │ │ t=1     │ │ t=T-1   │
   └────┬────┘ └────┬────┘ └────┬────┘
        │          │           │
        └──────────┼───────────┘
                   │  context (all demos → current)
                   ▼
             ┌─────────┐
             │ Current │
             │ t=T     │
             └─────────┘
```

**3. ACTION EDGES (current → future predictions)**
```
             ┌─────────┐
             │ Current │
             │ t=T     │
             └────┬────┘
                  │  current → action
                  ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Action  │ │ Action  │ │ Action  │
   │ t=T+1   │ │ t=T+2   │ │ t=T+P   │
   └─────────┘ └─────────┘ └─────────┘
        ◄──────────────────────►
              action temporal
```

**4. SCENE-GRIPPER EDGES (spatial relationship)**
```
   ┌─────────────────┐
   │ Scene Points    │  (16 PointNet++ output nodes)
   │ ● ● ● ● ● ● ... │
   └────────┬────────┘
            │ scene → gripper
            ▼
   ┌─────────────────┐
   │ Gripper Keypts  │  (6 keypoints defining gripper shape)
   │ ● ● ● ● ● ●     │
   └─────────────────┘
```

**5. CROSS-ARM EDGES (coordination between left and right)**
```
        LEFT ARM                              RIGHT ARM
   ┌─────────────────┐                   ┌─────────────────┐
   │ Gripper (demo)  │ ◄════ cross ════► │ Gripper (demo)  │
   └─────────────────┘      arm          └─────────────────┘
   
   ┌─────────────────┐                   ┌─────────────────┐
   │ Gripper (curr)  │ ◄════ cross ════► │ Gripper (curr)  │
   └─────────────────┘      arm          └─────────────────┘
   
   ┌─────────────────┐                   ┌─────────────────┐
   │ Gripper (action)│ ◄════ cross ════► │ Gripper (action)│
   └─────────────────┘      arm          └─────────────────┘
```

### 4.4 Edge Types: Complete List

```python
# Total: 18 edge types when cross-arm is enabled

# ========== LOCAL SPATIAL (local_encoder) ==========
('gripper_left', 'rel', 'gripper_left'),    # Left self-attention
('gripper_right', 'rel', 'gripper_right'),  # Right self-attention

# ========== DEMO TEMPORAL (cond_encoder) ==========
('gripper_left', 'demo', 'gripper_left'),   # t → t-1 within demo
('gripper_right', 'demo', 'gripper_right'),

# ========== CONTEXT: Demo → Current (cond_encoder) ==========
('gripper_left', 'cond', 'gripper_left'),   # All demo → current
('gripper_right', 'cond', 'gripper_right'),

# ========== SCENE-SCENE (cond_encoder + action_encoder) ==========
('scene_left', 'rel_demo', 'scene_left'),   # Demo scene self
('scene_right', 'rel_demo', 'scene_right'),
('scene_left', 'rel_action', 'scene_left'), # Action scene self
('scene_right', 'rel_action', 'scene_right'),

# ========== SCENE-GRIPPER (cond_encoder + action_encoder) ==========
('scene_left', 'rel_demo', 'gripper_left'),   # Demo scene → demo grip
('scene_right', 'rel_demo', 'gripper_right'),
('scene_left', 'rel_action', 'gripper_left'), # Action scene → action grip
('scene_right', 'rel_action', 'gripper_right'),

# ========== ACTION TEMPORAL (action_encoder) ==========
('gripper_left', 'time_action', 'gripper_left'),   # Action → action
('gripper_right', 'time_action', 'gripper_right'),
('gripper_left', 'rel_cond', 'gripper_left'),      # Current → action
('gripper_right', 'rel_cond', 'gripper_right'),

# ========== CROSS-ARM (cond_encoder + action_encoder) ==========
('gripper_left', 'cross', 'gripper_right'),        # Current left → current right
('gripper_right', 'cross', 'gripper_left'),
('gripper_left', 'cross_demo', 'gripper_right'),   # Demo left ↔ demo right
('gripper_right', 'cross_demo', 'gripper_left'),
('gripper_left', 'cross_action', 'gripper_right'), # Action left ↔ action right
('gripper_right', 'cross_action', 'gripper_left'),
```

### 4.5 Edge Connectivity Rules

**Demo Temporal (`demo`):**
```python
# Connects gripper t to gripper t-1 within same demo
mask = (
    same_batch &
    same_demo &
    src_time < T &         # Both in demo range
    dst_time < T &
    (dst_time - src_time) == -1  # Consecutive, backwards looking
)
```

**Context (`cond`):**
```python
# All demo grippers connect to current gripper
mask = (
    same_batch &
    src_time < T &     # Source is demo
    dst_time == T      # Dest is current
)
```

**Cross Demo (`cross_demo`):**
```python
# IMPORTANT: Full trajectory visibility (no time window!)
mask = (
    same_batch &
    same_demo &
    src_time < T &     # Both in demo
    dst_time < T
    # NO time restriction - left at t=0 sees right at t=9
)
```
This is a **critical design decision**: Full temporal visibility enables learning long-range coordination patterns.

**Cross Action (`cross_action`):**
```python
# Same action timestep only
mask = (
    same_batch &
    src_time > T &    # Both in action range
    dst_time > T &
    src_time == dst_time  # Same timestep
)
```

### 4.6 Edge Attributes: Fourier-Encoded Relative Geometry

All edges carry **relative geometric information** encoded using **Fourier positional encoding**.

**Why Fourier Encoding?**
- Raw 3D vectors are low-dimensional and hard for networks to learn fine distinctions
- Fourier encoding lifts 3D → ~63D (with 10 frequencies: `3 + 3×10×2 = 63`)
- Creates multi-scale representations that help transformers learn spatial patterns

```python
# PositionalEncoder: 3D → 63D via sine/cosine at multiple frequencies
def fourier_encode(x):  # x: [N, 3]
    # Returns: [x, sin(x), cos(x), sin(2x), cos(2x), ..., sin(512x), cos(512x)]
    # Output shape: [N, 63]
```

**Edge Attribute Computation (for gripper nodes):**

```python
# Given: relative transform T between source and destination nodes
# T = T_src_to_world @ T_world_to_dst (transforms points from dst frame to src frame)

# 1. TRANSLATION: Extract the translation part
relative_translation = T[:, :3, 3]  # [N, 3]

# 2. ROTATION AS POSITION DELTA: How does the rotation move the source position?
#    Instead of encoding rotation directly, we encode its EFFECT on position
rotated_src_pos = T[:, :3, :3] @ src_pos       # Apply rotation to source position
rotation_effect = rotated_src_pos - src_pos    # Delta caused by rotation [N, 3]

# Fourier encode both and concatenate
edge_attr = concatenate([
    fourier_encode(relative_translation),  # ~63D
    fourier_encode(rotation_effect)        # ~63D  
])  # Total: ~126D per edge
```

**Why encode rotation as position delta (not as angles or 6D)?**
- This is a **keypoint-based representation**: each gripper has 6 keypoints
- Rotation is captured by how those keypoints move
- Avoids singularities of angle representations (gimbal lock, discontinuities)
- Matches how actions are predicted (also as keypoint deltas)

**For scene nodes** (no orientation, just positions):
```python
relative_position = dst_pos - src_pos
edge_attr = concatenate([
    fourier_encode(relative_position),
    fourier_encode(relative_position)  # Repeated (no rotation info available)
])
```

**For cross-arm edges** (left ↔ right):
```python
# T_left_to_right transforms points from right frame to left frame
dst_in_src_frame = T[:, :3, :3] @ dst_pos + T[:, :3, 3]  # Where right appears in left's frame
relative_position = dst_in_src_frame - src_pos

rotation_effect = T[:, :3, :3] @ src_pos - src_pos  # Same pattern

edge_attr = concatenate([
    fourier_encode(relative_position),
    fourier_encode(rotation_effect)
])
```

**Key insight**: All edge attributes use the same two-part encoding:
1. **Relative position** (where is the other node?)
2. **Rotation effect** (how would rotation move my position?)

This is consistent with the keypoint-based action prediction and maintains SE(3) equivariance properties.

### 4.7 Gripper Keypoints

The gripper is represented by 6 keypoints:

```python
gripper_keypoints = torch.tensor([
    [0., 0., 0.],      # Center of gripper (origin)
    [0., 0., -0.03],   # Tail (back of gripper)
    [0., 0.03, 0.],    # Side (left)
    [0., -0.03, 0.],   # Side (right)
    [0., 0.03, 0.03],  # Finger tip (left)
    [0., -0.03, 0.03], # Finger tip (right)
]) * 2  # Scaled by 2
```

These 6 points define the gripper geometry. When we predict actions, we predict where these keypoints should move, then solve for the rigid transform.

### 4.8 Embedding Index System

**The Problem**: In naive implementations, demo/current/action gripper nodes might use the same embedding indices, causing them to be indistinguishable.

**The Solution**: Non-overlapping index ranges with temporal identity:

```python
# For num_demos (D), trajectory_horizon (T), gripper_keypoints (G), prediction_horizon (P):

# Demo embeddings: Encode (demo_index, timestep, keypoint_index)
# Range: [0, num_demos * trajectory_horizon * gripper_keypoints - 1]
gripper_embedding_demo = demo_index * trajectory_horizon * gripper_keypoints + timestep * gripper_keypoints + keypoint_index

# Current embeddings: Start AFTER demo range
# Range: [num_demos*traj_horizon*keypoints, num_demos*traj_horizon*keypoints + keypoints - 1]
num_demo_embeddings = num_demos * trajectory_horizon * gripper_keypoints
gripper_embedding_current = keypoint_index + num_demo_embeddings

# Action embeddings: Start AFTER current
# Range continues after current embeddings
gripper_embedding_action = keypoint_index + gripper_keypoints * (action_timestep - trajectory_horizon - 1) + num_demo_embeddings + gripper_keypoints
```

This ensures:
- Demo at t=0 has different embedding than t=5
- Current has different embedding than any demo
- Action at step 0 has different embedding than step 7

---

## 5. Model Architecture In Detail

**File**: `src/models/model.py` (483 lines)

### 5.1 BimanualAGI Class Structure

```python
class BimanualAGI(nn.Module):
    def __init__(self, config):
        # Scene Encoders
        self.scene_encoder = SceneEncoder(...)  # PointNet++
        # If not shared:
        # self.scene_encoder_right = SceneEncoder(...)
        
        # Graph Representation
        self.graph = BimanualGraphRep(config)
        
        # Three-Stage Graph Transformers
        self.local_encoder = GraphTransformer(...)  # 2 layers
        self.cond_encoder = GraphTransformer(...)   # 2 layers
        self.action_encoder = GraphTransformer(...) # 2 layers
        
        # Coordination Gate (learnable)
        self.coordination_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Prediction Heads (per arm)
        self.pred_head_trans_left = MLP([hidden_dim, 512, 3])
        self.pred_head_rot_left = MLP([hidden_dim, 512, 3])
        self.pred_head_grip_left = MLP([hidden_dim, 512, 1])
        # ... same for right
```

### 5.2 Scene Encoder Details

Uses the same **PointNet++** architecture as original IP:

```python
class SceneEncoder(nn.Module):
    # Two set abstraction layers with grouping
    self.sa1_module = SAModule(...)  # Global features
    self.sa2_module = SAModule(...)  # Local refinement
    
    def forward(self, features, pos, batch):
        # Input: pos [N, 3], batch [N]
        # Output: embeddings [S, embd_dim], pooled_pos [S, 3]
        x, pos, batch = self.sa1_module(features, pos, batch)
        x, pos, batch = self.sa2_module(x, pos, batch)
        return x, pos, batch  # [S, embd_dim] per sample
```

**Key Config:**
- `num_freqs`: 10 (positional encoding frequencies)
- `embd_dim`: 512 (local_nn_dim)
- Output: 16 scene nodes per arm per timestep

### 5.3 Three-Stage Graph Transformer

**Stage 1: Local Encoder**
- Edges: `rel` only (gripper self-edges)
- Purpose: Understand spatial relations within each arm independently
- No cross-arm: Gating hasn't been computed yet!

**Stage 2: Cond Encoder**
- Edges: `demo`, `cond`, `rel_demo`, `cross`, `cross_demo`
- Purpose: Propagate demonstration context to current state
- Cross-arm enabled: After gating scales edge attributes

**Stage 3: Action Encoder**
- Edges: `time_action`, `rel_cond`, `rel_action`, `cross_action`
- Purpose: Predict future actions based on context
- Cross-arm enabled: Coordinated action prediction

Each stage uses:
```python
GraphTransformer(
    in_channels=hidden_dim,      # 1024
    hidden_channels=hidden_dim,  # 1024
    heads=hidden_dim // 64,      # 16 heads
    num_layers=2,                # 2 transformer layers
    edge_dim=edge_dim,           # from PositionalEncoder
    dropout=0.0,
    norm='layer'
)
```

### 5.4 Selective Coordination Gating

**Motivation**: Not all bimanual tasks need the same level of coordination:
- Symmetric lift: High coordination (arms must move together)
- Independent pick-and-place: Low coordination (arms ignore each other)
- Handover: Medium coordination (timing matters, but not tight coupling)

**Implementation:**

```python
def forward(self, data):
    # After local_encoder processes independently...
    x_dict = self.local_encoder(...)
    
    # Pool current gripper features
    left_curr_mask = self.graph.graph.gripper_left_time == T
    right_curr_mask = self.graph.graph.gripper_right_time == T
    
    left_pooled = x_dict['gripper_left'][left_curr_mask].mean(dim=0)
    right_pooled = x_dict['gripper_right'][right_curr_mask].mean(dim=0)
    
    # Compute gate: [0, 1]
    combined = torch.cat([left_pooled, right_pooled], dim=-1)
    coord_gate = self.coordination_gate(combined)  # Sigmoid output
    
    # Scale ALL cross-arm edge attributes
    for edge_type in edge_attr_dict.keys():
        if 'cross' in edge_type[1]:
            edge_attr_dict[edge_type] *= coord_gate
    
    # Now cond_encoder and action_encoder see scaled cross-arm edges
    x_dict = self.cond_encoder(x_dict, ...)
    x_dict = self.action_encoder(x_dict, ...)
```

**Expected Behavior:**
- Independent tasks: gate → 0.1-0.3 (weak cross-arm attention)
- Coordinated tasks: gate → 0.7-1.0 (strong cross-arm attention)

The gate is learned end-to-end; no explicit supervision is needed.

### 5.5 Prediction Heads

After graph processing, action node features are extracted and passed through per-arm prediction heads:

```python
# Extract action gripper nodes (nodes with time > trajectory_horizon are action nodes)
trajectory_horizon = self.traj_horizon
left_action_mask = self.graph.graph.gripper_left_time > trajectory_horizon
left_gripper_features = x_dict['gripper_left'][left_action_mask].view(
    batch_size, prediction_horizon, gripper_keypoints, hidden_dim
)

# Predict for each keypoint
translation_delta_left = self.pred_head_trans_left(left_gripper_features)   # [batch, pred_horizon, keypoints, 3]
rotation_delta_left = self.pred_head_rot_left(left_gripper_features)         # [batch, pred_horizon, keypoints, 3]
gripper_state_left = self.pred_head_grip_left(left_gripper_features)         # [batch, pred_horizon, keypoints, 1]

predictions_left = cat([translation_delta_left, rotation_delta_left, gripper_state_left], dim=-1)  # [batch, pred_horizon, keypoints, 7]
```

**What the 7 outputs mean:**
- `[0:3]`: Translation delta (how far the entire gripper should move)
- `[3:6]`: Rotation delta (as keypoint offsets: where each keypoint moves due to rotation)
- `[6]`: Gripper state (open/closed)

### 5.6 Label Computation (get_labels)

During training, we need supervision labels. The model predicts **deltas** from noisy to ground-truth:

```python
def get_labels(self, ground_truth_actions, noisy_actions, ground_truth_grips, noisy_grips, arm):
    # ground_truth_actions, noisy_actions: [batch_size, prediction_horizon, 4, 4] SE(3) transforms
    
    # Compute relative transform: transform_delta = transform_noisy^{-1} @ transform_ground_truth
    transform_world_to_noisy = noisy_actions.view(-1, 4, 4)
    transform_noisy_to_world = torch.inverse(transform_world_to_noisy)
    transform_world_to_gt = ground_truth_actions.view(-1, 4, 4)
    transform_noisy_to_gt = torch.bmm(transform_noisy_to_world, transform_world_to_gt)  # Delta from noisy to ground truth
    
    # Translation labels: Just the translation part of delta
    translation_labels = transform_noisy_to_gt[..., :3, 3]  # [batch_size, prediction_horizon, 3]
    
    # Rotation labels: As keypoint offsets
    # Transform base keypoints by rotation-only delta
    rotation_only_transform = transform_noisy_to_gt.clone()
    rotation_only_transform[..., :3, 3] = 0  # Zero out translation
    rotated_keypoints = transform(gripper_keypoints, rotation_only_transform)
    rotation_labels = rotated_keypoints - gripper_keypoints  # [batch, pred_horizon, keypoints, 3]
    
    # Expand translation to per-keypoint
    translation_labels = translation_labels[:, :, None, :].expand(-1, -1, gripper_keypoints, -1)
    
    return cat([translation_labels, rotation_labels, ground_truth_grips], dim=-1)  # [batch, pred_horizon, keypoints, 7]
```

---

## 6. Diffusion Process

**File**: `src/models/diffusion.py` (515 lines)

### 6.1 Coupled Diffusion Overview

Both arms undergo diffusion at the **same timestep** using the **same noise scheduler**:

```python
def training_step(self, data, batch_idx):
    # Sample ONE set of diffusion timesteps for BOTH arms
    diffusion_timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
    
    # Add noise to both arms using SAME timesteps
    noisy_actions_left, noisy_grips_left = self.add_noise(
        data.actions_left, data.actions_grip_left, diffusion_timesteps
    )
    noisy_actions_right, noisy_grips_right = self.add_noise(
        data.actions_right, data.actions_grip_right, diffusion_timesteps
    )
```

This coupling ensures the model learns to predict coherent bimanual actions.

### 6.2 Noise Addition Process

```python
def add_noise(self, actions, gripper_actions, diffusion_timesteps, normalizer):
    # actions: [batch_size, prediction_horizon, 4, 4] SE(3) transforms
    
    # 1. Convert SE(3) to 6D (translation + angle-axis)
    actions_6d = transforms_to_actions(actions)  # [batch_size, prediction_horizon, 6]
    
    # 2. Normalize to [-1, 1]
    actions_6d_normalized = normalizer.normalize_actions(actions_6d)
    
    # 3. Add scheduler noise
    noise = torch.randn_like(actions_6d_normalized)
    noisy_actions_normalized = self.noise_scheduler.add_noise(actions_6d_normalized, noise, diffusion_timesteps)
    noisy_actions_normalized = torch.clamp(noisy_actions_normalized, -1, 1)
    
    # 4. Denormalize back
    noisy_actions_6d = normalizer.denormalize_actions(noisy_actions_normalized)
    
    # 5. Convert back to SE(3) transforms
    noisy_actions_se3 = actions_to_transforms(noisy_actions_6d)  # [batch_size, prediction_horizon, 4, 4]
    
    return noisy_actions_se3, noisy_gripper_actions
```

### 6.3 Training Step: Complete Flow

```python
def training_step(self, data, batch_idx):
    # 1. Sample diffusion timesteps
    diffusion_timesteps = torch.randint(0, 100, (batch_size,))
    
    # 2. Add noise to both arms
    noisy_actions_left, noisy_grips_left = self.add_noise(...)
    noisy_actions_right, noisy_grips_right = self.add_noise(...)
    
    # 3. Compute labels (what model should predict)
    labels_left = self.model.get_labels(
        ground_truth_actions_left, noisy_actions_left, ground_truth_grips_left, noisy_grips_left, 'left'
    )
    labels_right = self.model.get_labels(...)
    
    # 4. Normalize labels
    labels_left[..., :6] = normalizer.normalize_labels(labels_left[..., :6])
    labels_right[..., :6] = normalizer.normalize_labels(...)
    
    # 5. CRITICAL: Save ground truth before overwriting
    ground_truth_left = data.actions_left.clone()
    ground_truth_right = data.actions_right.clone()
    
    # 6. Store noisy actions in data (this is what model sees)
    data.actions_left = noisy_actions_left
    data.actions_right = noisy_actions_right
    data.diffusion_timestep = diffusion_timesteps
    
    # 7. Forward pass
    predictions_left, predictions_right = self.model(data)
    
    # 8. Compute loss
    loss_left = L1_loss(predictions_left, labels_left)
    loss_right = L1_loss(predictions_right, labels_right)
    total_loss = (loss_left + loss_right) / 2
    
    # 9. Optional: Coordination loss
    if use_coordination_loss:
        decoded_actions_left = decode_predictions(predictions_left, noisy_actions_left)
        decoded_actions_right = decode_predictions(predictions_right, noisy_actions_right)
        coordination_loss = compute_coordination_loss(decoded_actions_left, decoded_actions_right, 
                                                       ground_truth_left, ground_truth_right)
        total_loss += coordination_weight * coordination_loss
    
    return total_loss
```

### 6.4 Coordination Consistency Loss

Encourages maintaining relative pose between arms:

```python
def _compute_coordination_loss(self, pred_left, pred_right, gt_left, gt_right):
    # Compute relative transform: T_left_to_right = T_left^-1 @ T_right
    T_rel_pred = torch.bmm(pred_left.inverse(), pred_right)
    T_rel_gt = torch.bmm(gt_left.inverse(), gt_right)
    
    # Loss on relative translation
    trans_loss = ||T_rel_pred[:,:,:3,3] - T_rel_gt[:,:,:3,3]||
    
    # Loss on relative rotation (Frobenius norm)
    rot_diff = T_rel_pred[:,:,:3,:3] - T_rel_gt[:,:,:3,:3]
    rot_loss = ||rot_diff||_F
    
    return trans_loss + rot_loss
```

> [!NOTE]
> The coordination loss must be computed on **decoded model predictions**, not on the noisy inputs. The training code first decodes predictions back to SE(3) actions before computing this loss.

### 6.5 Inference: DDIM Denoising

```python
def test_step(self, data, batch_idx):
    # 1. Start from pure noise
    noisy_left = randn([B, P, 4, 4])  # Converted from 6D noise
    noisy_right = randn([B, P, 4, 4])
    noisy_grips_left = randn([B, P, 1])
    noisy_grips_right = randn([B, P, 1])
    
    # 2. Set DDIM schedule (8 steps default)
    self.noise_scheduler.set_timesteps(8)
    # timesteps = [875, 750, 625, 500, 375, 250, 125, 0] (approximate)
    
    # 3. Iterative denoising
    for k in range(7, -1, -1):  # k = 7, 6, 5, ..., 0
        # Update data with current noisy actions
        data.actions_left = noisy_left
        data.actions_right = noisy_right
        data.diff_time = timesteps[k]
        
        # Predict deltas
        preds_left, preds_right = self.model(data)
        
        # Denormalize predictions
        preds_left[..., :6] = denormalize(preds_left[..., :6])
        preds_right[..., :6] = denormalize(...)
        
        # Apply diffusion step
        noisy_left, noisy_grips_left = self._diffusion_step(
            noisy_left, noisy_grips_left, preds_left, k, 'left'
        )
        noisy_right, noisy_grips_right = self._diffusion_step(...)
    
    # 4. Return final actions
    return noisy_left, sign(noisy_grips_left), noisy_right, sign(noisy_grips_right)
```

### 6.6 Diffusion Step: Delta Application

```python
def _diffusion_step(self, noisy_actions, noisy_grips, preds, k, arm):
    # Get current gripper keypoint positions
    current_kp = transform(gripper_keypoints, noisy_actions)  # [B, P, G, 3]
    
    # Predict target keypoint positions
    pred_kp = current_kp + preds[..., 3:6] + mean(preds[..., :3])
    
    # DDIM scheduler step for positions
    new_kp = noise_scheduler.step(
        model_output=pred_kp,
        sample=current_kp,
        timestep=k
    ).prev_sample
    
    # Solve for rigid transform: current_kp → new_kp
    T_delta = get_rigid_transforms(current_kp, new_kp)  # Kabsch algorithm
    
    # Compose with current noisy action
    noisy_actions = noisy_actions @ T_delta
    
    # Clamp to valid action range
    noisy_actions = normalize → clamp → denormalize → to_SE3
    
    return noisy_actions, noisy_grips
```

---

## 7. Pseudo-Demonstration Generation

**Files**: `generator.py`, `patterns/coordination_patterns.py`, `objects.py`

### 7.1 Why Pseudo-Demonstrations?

Training data for bimanual tasks is scarce. Pseudo-demonstrations are procedurally generated:
- No robot required
- Infinite diversity
- Cover all coordination patterns
- Include perturbations for robustness

### 7.2 Four Coordination Patterns

| Pattern               | Weight | Description                                  |
| --------------------- | ------ | -------------------------------------------- |
| `symmetric_lift`      | 30%    | Both arms approach from sides, lift together |
| `handover`            | 25%    | One picks, moves to middle, other receives   |
| `hold_and_manipulate` | 25%    | One holds, other manipulates                 |
| `independent`         | 20%    | Both work on separate objects                |

### 7.3 Pattern Generators

Each pattern is implemented as a class:

```python
class SymmetricLiftPattern(PatternGenerator):
    def generate(self, scene):
        T = self.trajectory_length  # 100 frames
        
        # Find target object
        target = scene.get_object_by_type('tray') or scene.objects[0]
        obj_pos = target.centroid
        obj_width = target.dimensions['width']
        
        # Compute grasp positions
        left_grasp = obj_pos + [0, obj_width/2 + 0.02, 0.05]
        right_grasp = obj_pos + [0, -obj_width/2 - 0.02, 0.05]
        
        # Define waypoints with timing
        # Phase 1: Approach (0-20%)
        # Phase 2: Grasp (20-30%)
        # Phase 3: Lift (30-70%)
        # Phase 4: Hold (70-100%)
        
        left_waypoints = [
            self.left_home,
            left_grasp + [0, 0, 0.1],  # Above
            left_grasp,                 # At grasp
            left_grasp,                 # Hold
            left_grasp + [0, 0, 0.15], # Lifted
            left_grasp + [0, 0, 0.15], # Hold lifted
        ]
        
        # Interpolate to full trajectory
        T_left = interpolate(left_waypoints, T)
        T_right = interpolate(right_waypoints, T)
        
        # Gripper timing: close at 30%
        grips_left = zeros(T)
        grips_left[int(0.30*T):] = 1.0
        grips_right = grips_left.copy()
        
        return T_left, T_right, grips_left, grips_right
```

### 7.4 Scene Generation

```python
class SceneGenerator:
    def __init__(self):
        self.object_generator = ProceduralObjectGenerator()
        self.type_weights = {
            'box': 0.25,
            'cylinder': 0.20,
            'sphere': 0.10,
            'tray': 0.10,
            'bottle': 0.15,
            'mug': 0.10,
            'plate': 0.10,
        }
    
    def generate(self, required_types=None):
        scene = Scene()
        num_objects = random.randint(1, 5)
        
        for _ in range(num_objects):
            obj_type = sample_weighted(self.type_weights)
            obj = self.object_generator.generate(obj_type)
            
            # Random position within workspace
            x = random.uniform(-0.3, 0.3)
            y = random.uniform(-0.3, 0.3)
            z = 0  # On table
            obj.set_position([x, y, z])
            
            scene.add_object(obj)
        
        return scene
```

Object types include:
- **Box**: Cuboid with random dimensions
- **Cylinder**: Variable radius/height
- **Sphere**: Variable radius
- **Tray**: Hollow box (good for symmetric lift)
- **Bottle**: Cylinder + neck
- **Mug**: Cylinder + handle
- **Plate**: Flat cylinder

### 7.5 Dynamic Point Cloud Rendering

**Critical Feature**: Objects move when grasped!

```python
class SceneRenderer:
    def __init__(self, scene):
        self.scene = scene
        self.grasped_by_left = None
        self.grasped_by_right = None
        self.left_grasp_offset = None  # Object pose in gripper frame
        self.right_grasp_offset = None
    
    def render(self, T_left, T_right, grip_left, grip_right,
               prev_grip_left, prev_grip_right, num_points=2048):
        
        # Detect grasp initiation (grip closing)
        if grip_left > 0.5 and prev_grip_left <= 0.5:
            obj = self._detect_grasp(T_left[:3, 3])
            if obj is not None:
                self.grasped_by_left = obj
                # Store offset: obj_pose_in_gripper_frame
                self.left_grasp_offset = inv(T_left) @ obj.pose
        
        # Detect release
        if grip_left <= 0.5 and prev_grip_left > 0.5:
            self.grasped_by_left = None
        
        # Update grasped object pose
        if self.grasped_by_left is not None:
            self.grasped_by_left.pose = T_left @ self.left_grasp_offset
        
        # Same for right arm...
        
        # Render combined point cloud
        all_points = []
        for obj in self.scene.objects:
            all_points.append(obj.world_points)
        
        points = concatenate(all_points)
        
        # Subsample to num_points
        if len(points) > num_points:
            indices = random.choice(len(points), num_points, replace=False)
            points = points[indices]
        
        return points
```

### 7.6 Data Augmentations

```python
def _augment(self, T_left, T_right, grips_left, grips_right, pattern_type):
    # 1. Timing Jitter (40% prob)
    if random() < 0.40:
        # Phase offset: ±10 frames
        shift = randint(-10, 11)
        T_right = shift_trajectory(T_right, shift)
        grips_right = shift_trajectory(grips_right, shift)
        
        # Time stretching (50% prob)
        if random() < 0.5:
            stretch = uniform(0.85, 1.15)
            T_right = time_stretch(T_right, stretch)
            grips_right = time_stretch(grips_right, stretch)
    
    # 2. Perturbations (30% prob)
    if random() < 0.30:
        # Add noise to mid-trajectory (forces recovery)
        perturb_idx = randint(T//3, 2*T//3)
        T_left[perturb_idx] += random_SE3_noise()
    
    # 3. Gripper Flips (10% prob)
    if random() < 0.10:
        flip_idx = randint(T//4, 3*T//4)
        grips_left[flip_idx:] = 1 - grips_left[flip_idx:]
    
    # 4. Arm Swaps (20% prob) - not for symmetric lift
    if random() < 0.20 and pattern_type != 'symmetric_lift':
        T_left, T_right = T_right, T_left
        grips_left, grips_right = grips_right, grips_left
    
    return T_left, T_right, grips_left, grips_right
```

### 7.7 Complete Generation Pipeline

```python
class BimanualPseudoDemoGenerator:
    def generate(self):
        # 1. Sample pattern type
        pattern_type = self._sample_pattern()  # weighted random
        
        # 2. Generate scene appropriate for pattern
        scene = self._generate_scene(pattern_type)
        
        # 3. Generate coordinated trajectories
        T_left, T_right, grips_left, grips_right = \
            self._generate_trajectories(pattern_type, scene)
        
        # 4. Apply augmentations
        T_left, T_right, grips_left, grips_right = \
            self._augment(T_left, T_right, grips_left, grips_right, pattern_type)
        
        # 5. Render point clouds at each timestep
        pcds = self._render_point_clouds(
            scene, T_left, T_right, grips_left, grips_right
        )
        
        # 6. Package as trajectory
        return BimanualTrajectory(
            T_w_left=T_left,
            T_w_right=T_right,
            grips_left=grips_left,
            grips_right=grips_right,
            pcds=pcds,
            coordination_type=pattern_type
        )
```

---

## 8. Dataset & Data Pipeline

**File**: `bimanual_dataset.py` (434 lines)

### 8.1 Data Structures

```python
@dataclass
class BimanualTrajectory:
    T_w_left: np.ndarray    # [T, 4, 4] left poses in world frame
    T_w_right: np.ndarray   # [T, 4, 4] right poses
    grips_left: np.ndarray  # [T] left gripper states (0=open, 1=closed)
    grips_right: np.ndarray # [T] right gripper states
    pcds: List[np.ndarray]  # [T] list of [N, 3] point clouds
    coordination_type: str  # Pattern name

@dataclass
class BimanualGraphData(Data):
    # Point clouds in arm frames
    pos_demos_left: Tensor   # [total_demo_points, 3]
    pos_demos_right: Tensor
    batch_demos_left: Tensor # [total_demo_points] batch indices
    batch_demos_right: Tensor
    
    pos_obs_left: Tensor     # [obs_points, 3]
    pos_obs_right: Tensor
    
    # Poses and actions
    demo_T_w_left: Tensor    # [B, D, T, 4, 4]
    demo_T_w_right: Tensor
    
    T_obs_left: Tensor       # [B, 4, 4]
    T_obs_right: Tensor
    
    actions_left: Tensor     # [B, P, 4, 4]
    actions_right: Tensor
    
    # Gripper states
    demo_grips_left: Tensor  # [B, D, T, 1]
    current_grip_left: Tensor # [B]
    actions_grip_left: Tensor # [B, P]
    # ... same for right
    
    # Diffusion
    diff_time: Tensor        # [B, 1]
```

### 8.2 BimanualRunningDataset

Generates data on-the-fly:

```python
class BimanualRunningDataset(Dataset):
    def __init__(self, config, buffer_size=10000):
        self.generator = BimanualPseudoDemoGenerator(config)
        self.buffer = []
        self._fill_buffer()  # Pre-generate 10K trajectories
    
    def _fill_buffer(self):
        print(f"Filling buffer with {buffer_size} trajectories...")
        for _ in tqdm(range(buffer_size)):
            self.buffer.append(self.generator.generate())
    
    def __getitem__(self, idx):
        trajectory = self.buffer[idx % len(self.buffer)]
        
        # Occasionally regenerate for diversity
        if random() < 0.05:
            self.buffer[idx % len(self.buffer)] = self.generator.generate()
        
        return self._trajectory_to_graph_data(trajectory)
    
    def _trajectory_to_graph_data(self, trajectory):
        # 1. Sample observation point in trajectory
        obs_idx = randint(T // 3, 2 * T // 3)
        
        # 2. Sample demonstration waypoints (before observation)
        demo_indices = np.linspace(0, obs_idx - 1, num_waypoints).astype(int)
        
        # 3. Sample future action indices
        future_indices = range(obs_idx, min(obs_idx + pred_horizon, len(traj)))
        
        # 4. Transform point clouds to arm frames
        for i in demo_indices:
            T_left_inv = np.linalg.inv(trajectory.T_w_left[i])
            demo_pcds_left.append(transform_pcd(trajectory.pcds[i], T_left_inv))
            
            T_right_inv = np.linalg.inv(trajectory.T_w_right[i])
            demo_pcds_right.append(transform_pcd(trajectory.pcds[i], T_right_inv))
        
        # 5. Compute actions (relative transforms)
        T_obs_left = trajectory.T_w_left[obs_idx]
        T_obs_right = trajectory.T_w_right[obs_idx]
        
        actions_left = []
        for future_i in future_indices:
            # action = T_obs^{-1} @ T_future
            action = np.linalg.inv(T_obs_left) @ trajectory.T_w_left[future_i]
            actions_left.append(action)
        
        # 6. Package everything
        return BimanualGraphData(
            pos_demos_left=...,
            actions_left=torch.tensor(actions_left),
            ...
        )
```

### 8.3 Custom Collate Function

```python
def collate_bimanual(batch: List[BimanualGraphData]):
    result = BimanualGraphData()
    
    # Stack regular tensors
    for key in ['T_obs_left', 'actions_left', ...]:
        result[key] = torch.stack([b[key] for b in batch], dim=0)
    
    # Handle point clouds with variable size
    for arm in ['left', 'right']:
        pos_demos = result[f'pos_demos_{arm}']
        
        # Shape can be [B, D, T, N, 3] or [B, T, N, 3]
        if len(pos_demos.shape) == 5:
            B, D, T, N, _ = pos_demos.shape
            flat = pos_demos.reshape(B * D * T * N, 3)
            batch_idx = torch.arange(B * D * T).repeat_interleave(N)
        elif len(pos_demos.shape) == 4:
            B, T, N, _ = pos_demos.shape
            flat = pos_demos.reshape(B * T * N, 3)
            batch_idx = torch.arange(B * T).repeat_interleave(N)
        
        result[f'pos_demos_{arm}'] = flat
        result[f'batch_demos_{arm}'] = batch_idx
    
    return result
```

---

## 9. Training Details

**File**: `scripts/train.py` (212 lines)

### 9.1 Training Loop

```python
# Setup
config = get_config()
model = BimanualGraphDiffusion(config)
dataset = BimanualRunningDataset(config, buffer_size=10000)
loader = DataLoader(dataset, batch_size=8, shuffle=True, 
                    collate_fn=collate_bimanual)

# Train
trainer = L.Trainer(
    max_epochs=1000,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=1.0,
    precision='16-mixed',
    val_check_interval=5000,
    log_every_n_steps=100,
)

trainer.fit(model, loader, val_loader)
```

### 9.2 Hyperparameters

| Parameter                   | Value | Notes                            |
| --------------------------- | ----- | -------------------------------- |
| `batch_size`                | 8     | Smaller due to memory (2 arms)   |
| `lr`                        | 1e-5  | Learning rate                    |
| `weight_decay`              | 1e-2  | AdamW weight decay               |
| `hidden_dim`                | 1024  | Graph transformer channels       |
| `local_nn_dim`              | 512   | Scene encoder output             |
| `num_layers`                | 2     | Layers per transformer stage     |
| `num_scenes_nodes`          | 16    | Scene nodes per arm              |
| `num_demos`                 | 1     | Demonstrations per sample        |
| `traj_horizon`              | 10    | Waypoints per demo               |
| `pred_horizon`              | 8     | Action prediction steps          |
| `num_diffusion_iters_train` | 100   | Training diffusion steps         |
| `num_diffusion_iters_test`  | 8     | Inference diffusion steps (DDIM) |

### 9.3 Loss Function

```python
# Primary loss: L1 on predictions
loss_left = L1Loss(preds_left, labels_left)
loss_right = L1Loss(preds_right, labels_right)
total_loss = (loss_left + loss_right) / 2

# Optional: Coordination consistency (weight=0.1)
if use_coordination_loss:
    coord_loss = coordination_loss(pred_left, pred_right, gt_left, gt_right)
    total_loss += 0.1 * coord_loss
```

### 9.4 Optimizer Configuration

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-2
)

# Optional: Cosine LR schedule
scheduler = get_scheduler(
    'cosine',
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=50000000
)
```

---

## 10. Configuration Reference

**File**: `external/ip/configs/bimanual_config.py`

```python
config = {
    # ========== Cross-Arm Attention ==========
    'use_cross_arm_attention': True,
    'use_coordination_gate': True,
    'use_coordination_loss': False,  # Optional
    'coordination_loss_weight': 0.1,
    
    # ========== Scene Encoding ==========
    'shared_scene_encoder': True,  # Share left/right weights
    'num_scenes_nodes': 16,        # Per arm
    'pre_trained_encoder': False,
    'freeze_encoder': False,
    
    # ========== Architecture ==========
    'local_nn_dim': 512,
    'hidden_dim': 1024,
    'num_layers': 2,
    'local_num_freq': 10,  # Positional encoding
    'pos_in_nodes': True,
    
    # ========== Demonstrations ==========
    'num_demos': 1,
    'num_demos_test': 1,
    'traj_horizon': 10,
    'pre_horizon': 8,
    
    # ========== Training ==========
    'batch_size': 8,
    'lr': 1e-5,
    'weight_decay': 1e-2,
    'num_diffusion_iters_train': 100,
    'num_diffusion_iters_test': 8,
    
    # ========== Action Normalization ==========
    'min_actions': [-0.01, -0.01, -0.01, -3°, -3°, -3°],
    'max_actions': [+0.01, +0.01, +0.01, +3°, +3°, +3°],
    
    # ========== Pseudo-Demo Generation ==========
    'pseudo_demo': {
        'min_objects': 1,
        'max_objects': 5,
        'num_points_per_object': 512,
        'total_scene_points': 2048,
        'trajectory_length': 100,
        'num_waypoints': 10,
        'pattern_weights': {
            'symmetric_lift': 0.30,
            'handover': 0.25,
            'hold_and_manipulate': 0.25,
            'independent': 0.20,
        },
        'perturbation_prob': 0.30,
        'grip_flip_prob': 0.10,
        'arm_swap_prob': 0.20,
        'timing_jitter_prob': 0.40,
    },
    
    # ========== Gripper Geometry ==========
    'gripper_keypoints': [
        [0., 0., 0.],      # Center
        [0., 0., -0.03],   # Back
        [0., 0.03, 0.],    # Side
        [0., -0.03, 0.],   # Side
        [0., 0.03, 0.03],  # Finger
        [0., -0.03, 0.03], # Finger
    ] * 2,  # Scaled
}
```

---

## 11. File-by-File Reference

### Main Source Files (`src/`)

| File                          | Lines | Purpose                                             |
| ----------------------------- | ----- | --------------------------------------------------- |
| `src/models/graph_rep.py`     | 989   | Graph structure, node/edge construction, attributes |
| `src/models/model.py`         | 483   | BimanualAGI model with three-stage transformer      |
| `src/models/diffusion.py`     | 515   | Coupled diffusion training/inference                |
| `src/data/dataset.py`         | 434   | Dataset classes and collation                       |
| `src/data/generator.py`       | 605   | Pseudo-demonstration generation                     |
| `src/data/data_structures.py` | 241   | BimanualTrajectory, BimanualGraphData, etc.         |
| `src/data/objects.py`         | 463   | Procedural object generation                        |
| `src/evaluation/eval.py`      | -     | RLBench evaluation script                           |

### Script Files (`scripts/`)

| File                              | Lines | Purpose              |
| --------------------------------- | ----- | -------------------- |
| `scripts/train.py`                | 212   | Training script      |
| `scripts/slurm_bimanual_train.sh` | 65    | SLURM job submission |

### Pattern Files

| File                                         | Lines | Purpose                 |
| -------------------------------------------- | ----- | ----------------------- |
| `src/data/patterns/coordination_patterns.py` | 443   | Four pattern generators |

### External IP Files (`external/ip/`)

| File                                      | Purpose                                  |
| ----------------------------------------- | ---------------------------------------- |
| `external/ip/models/graph_rep.py`         | Original single-arm graph representation |
| `external/ip/models/model.py`             | Original single-arm AGI model            |
| `external/ip/models/scene_encoder.py`     | PointNet++ scene encoder (shared)        |
| `external/ip/models/graph_transformer.py` | Graph transformer layer                  |
| `external/ip/utils/common_utils.py`       | SE(3) utilities, rigid transforms        |
| `external/ip/utils/normalizer.py`         | Action normalization                     |
| `external/ip/configs/bimanual_config.py`  | Bimanual configuration                   |

---

## Summary: Key Design Principles

1. **Dual Egocentric Frames**: Each arm sees world from its own perspective → maintains spatial invariance

2. **Relative Cross-Arm Encoding**: SE(3) relative transforms between arms → base-motion invariance

3. **Three-Stage Transformer**: Local → Context → Action processing with proper edge assignment

4. **Selective Coordination Gating**: Learned gate scales cross-arm attention → task-appropriate coupling

5. **Full Demo Cross-Arm Visibility**: No time window → learn long-range coordination patterns

6. **Coupled Diffusion**: Same timestep for both arms → coherent bimanual predictions

7. **Non-Overlapping Embedding Indices**: Demo, current, and action nodes have distinct embedding ranges with temporal identity

8. **Diverse Pseudo-Demos**: Four coordination patterns + augmentations → robust generalization

9. **Dynamic Point Cloud Rendering**: Objects move when grasped → realistic scene evolution

---

*This document is comprehensive but can be extended. For implementation-level details, refer to the source files directly.*
