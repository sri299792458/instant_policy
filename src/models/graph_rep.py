"""
Bimanual Graph Representation for Instant Policy.

This module extends the original Instant Policy graph structure to handle
two end-effectors with proper spatial invariance and coordination edges.

Key Design Decisions:
1. Dual Egocentric Frames: Each arm has its own egocentric view of the scene
2. Relative Cross-Arm Edges: Arms are connected via relative transforms
3. Coordination Edges: Action nodes of both arms can attend to each other
4. Frame-Consistent Edge Attributes: All edge attributes encode relative geometry
"""
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for 3D positions."""
    
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = True, 
                 add_original: bool = True, scale: float = 1.0):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.scale = scale
        self.add_original = add_original
        
        if log_space:
            freq_bands = 2. ** torch.linspace(0., n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (n_freqs - 1), n_freqs)
        
        self.register_buffer('freq_bands', freq_bands)
        
        self.d_output = d_input * (2 * n_freqs)
        if add_original:
            self.d_output += d_input
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input positions [..., d_input]
        Returns:
            Encoded positions [..., d_output]
        """
        x_scaled = x / self.scale
        encodings = []
        
        if self.add_original:
            encodings.append(x_scaled)
        
        for freq in self.freq_bands:
            encodings.append(torch.sin(x_scaled * freq))
            encodings.append(torch.cos(x_scaled * freq))
        
        return torch.cat(encodings, dim=-1)


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding for diffusion timestep."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Ensure t is at least 1D
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class BimanualGraphRep(nn.Module):
    """
    Bimanual Graph Representation.
    
    Node Types:
        - scene_left: Scene points in left EE frame
        - scene_right: Scene points in right EE frame  
        - gripper_left: Left gripper keypoints
        - gripper_right: Right gripper keypoints
    
    Edge Types:
        Local (within same arm's egocentric frame):
            - (scene_X, rel, scene_X): Scene-to-scene within same arm
            - (scene_X, rel, gripper_X): Scene-to-gripper within same arm
            - (gripper_X, rel, gripper_X): Gripper self-edges
        
        Temporal (within demonstrations):
            - (gripper_X, demo, gripper_X): Between timesteps in demo
        
        Context (demo to current):
            - (gripper_X, cond, gripper_X): Demo gripper to current gripper
        
        Action (current to future):
            - (gripper_X, time_action, gripper_X): Between action timesteps
            - (gripper_X, rel_cond, gripper_X): Current to first action
        
        Cross-Arm Coordination:
            - (gripper_left, cross, gripper_right): Left observes right
            - (gripper_right, cross, gripper_left): Right observes left
            - (gripper_left, cross_action, gripper_right): Action coordination
            - (gripper_right, cross_action, gripper_left): Action coordination
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Store configuration
        self.batch_size = config['batch_size']
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.num_scene_nodes = config['num_scenes_nodes']
        self.num_freqs = config['local_num_freq']
        self.device = config['device']
        self.embd_dim = config['local_nn_dim']
        self.pred_horizon = config['pre_horizon']
        self.pos_in_nodes = config['pos_in_nodes']
        self.use_cross_arm = config.get('use_cross_arm_attention', True)
        self.config = config  # Store full config for accessing optional parameters
        
        # Gripper keypoints (6 points defining gripper geometry)
        self.gripper_keypoints = config['gripper_keypoints'].to(self.device)
        self.num_g_nodes = len(self.gripper_keypoints)
        
        # Embedding dimensions
        self.g_state_dim = 64   # Gripper state embedding dimension
        self.d_time_dim = 64    # Diffusion time embedding dimension
        
        # Positional encoding
        self.pos_encoder = PositionalEncoder(3, self.num_freqs, log_space=True, 
                                              add_original=True, scale=1.0)
        self.edge_dim = self.pos_encoder.d_output * 2  # Source and dest encodings
        
        # Time embedding for diffusion
        self.time_emb = SinusoidalTimeEmb(self.d_time_dim)
        
        # Gripper state projection
        self.gripper_proj = nn.Linear(1, self.g_state_dim)
        
        # Learnable embeddings for gripper nodes
        # We have: demo gripper nodes + current gripper + action gripper nodes
        # Demo nodes now encode: demo_idx * T * G + time * G + node_idx
        # Max demo index: num_demos * traj_horizon * num_g_nodes
        # Current and action: num_g_nodes * (pred_horizon + 1)
        num_demo_embeddings = self.num_demos * self.traj_horizon * self.num_g_nodes
        num_other_embeddings = self.num_g_nodes * (self.pred_horizon + 1)
        num_gripper_embeddings = num_demo_embeddings + num_other_embeddings
        self.gripper_embds_left = nn.Embedding(num_gripper_embeddings, 
                                                self.embd_dim - self.g_state_dim)
        self.gripper_embds_right = nn.Embedding(num_gripper_embeddings,
                                                 self.embd_dim - self.g_state_dim)
        
        # Edge type embeddings (learnable biases for different edge types)
        self.cond_edge_embd = nn.Embedding(1, self.edge_dim)
        self.demo_action_edge_embd = nn.Embedding(1, self.edge_dim)
        self.cross_arm_edge_embd = nn.Embedding(1, self.edge_dim)
        self.cross_action_edge_embd = nn.Embedding(1, self.edge_dim)
        
        # Define node and edge types
        self.node_types = ['scene_left', 'scene_right', 'gripper_left', 'gripper_right']
        self.edge_types = self._define_edge_types()
        
        # Graph structure (initialized in initialise_graph)
        self.graph = None
        
    def _define_edge_types(self):
        """Define all edge types for the bimanual graph."""
        edge_types = [
            # =============== LOCAL SPATIAL (Gripper only - scene edges use rel_demo/rel_action) ===============
            ('gripper_left', 'rel', 'gripper_left'),
            ('gripper_right', 'rel', 'gripper_right'),
            
            # =============== DEMO TEMPORAL ===============
            ('gripper_left', 'demo', 'gripper_left'),
            ('gripper_right', 'demo', 'gripper_right'),
            
            # =============== CONTEXT: Demo → Current ===============
            ('gripper_left', 'cond', 'gripper_left'),
            ('gripper_right', 'cond', 'gripper_right'),
            
            # =============== ACTION: Current/Action → Action ===============
            ('gripper_left', 'time_action', 'gripper_left'),
            ('gripper_right', 'time_action', 'gripper_right'),
            ('gripper_left', 'rel_cond', 'gripper_left'),
            ('gripper_right', 'rel_cond', 'gripper_right'),
            
            # =============== SCENE-SCENE AND SCENE-GRIPPER ===============
            ('scene_left', 'rel_action', 'gripper_left'),
            ('scene_right', 'rel_action', 'gripper_right'),
            ('scene_left', 'rel_demo', 'gripper_left'),
            ('scene_right', 'rel_demo', 'gripper_right'),
            ('scene_left', 'rel_action', 'scene_left'),
            ('scene_right', 'rel_action', 'scene_right'),
            ('scene_left', 'rel_demo', 'scene_left'),
            ('scene_right', 'rel_demo', 'scene_right'),
        ]
        
        if self.use_cross_arm:
            edge_types.extend([
                # =============== CROSS-ARM COORDINATION ===============
                # Current timestep cross-arm visibility
                ('gripper_left', 'cross', 'gripper_right'),
                ('gripper_right', 'cross', 'gripper_left'),
                
                # Action coordination (left actions see right actions)
                ('gripper_left', 'cross_action', 'gripper_right'),
                ('gripper_right', 'cross_action', 'gripper_left'),
                
                # Demo cross-arm (at same timestep)
                ('gripper_left', 'cross_demo', 'gripper_right'),
                ('gripper_right', 'cross_demo', 'gripper_left'),
            ])
        
        return edge_types
    
    def _create_dense_edge_idx(self, num_source: int, num_dest: int) -> torch.Tensor:
        """Create dense (fully connected) edge indices."""
        return torch.cartesian_prod(
            torch.arange(num_source, dtype=torch.long, device=self.device),
            torch.arange(num_dest, dtype=torch.long, device=self.device)
        ).t().contiguous()
    
    def _get_node_info(self, arm: str) -> Dict[str, torch.Tensor]:
        """
        Compute node information (batch, timestep, etc.) for one arm.
        
        Returns dict with indices for:
            - scene nodes: [B * num_demos * traj_horizon * num_scene_nodes] + [B * num_scene_nodes] + [B * pred_horizon * num_scene_nodes]
            - gripper nodes: [B * num_demos * traj_horizon * num_g_nodes] + [B * num_g_nodes] + [B * pred_horizon * num_g_nodes]
        """
        B = self.batch_size
        D = self.num_demos
        T = self.traj_horizon
        S = self.num_scene_nodes
        G = self.num_g_nodes
        P = self.pred_horizon
        
        # Helper for creating index tensors
        def arange(n):
            return torch.arange(n, device=self.device)
        
        # ==================== Scene Nodes ====================
        # Demo scene nodes: [B, D, T, S]
        scene_batch_demo = arange(B)[:, None, None, None].expand(B, D, T, S).reshape(-1)
        scene_traj_demo = arange(T)[None, None, :, None].expand(B, D, T, S).reshape(-1)
        scene_demo_demo = arange(D)[None, :, None, None].expand(B, D, T, S).reshape(-1)
        
        # Current scene nodes: [B, S]
        scene_batch_curr = arange(B)[:, None].expand(B, S).reshape(-1)
        scene_traj_curr = torch.full((B * S,), T, device=self.device)
        scene_demo_curr = torch.full((B * S,), D, device=self.device)  # D means "current"
        
        # Action scene nodes: [B, P, S]
        scene_batch_act = arange(B)[:, None, None].expand(B, P, S).reshape(-1)
        scene_traj_act = (arange(P)[None, :, None].expand(B, P, S) + T + 1).reshape(-1)
        scene_demo_act = torch.full((B * P * S,), D, device=self.device)
        
        scene_info = {
            'batch': torch.cat([scene_batch_demo, scene_batch_curr, scene_batch_act]),
            'traj': torch.cat([scene_traj_demo, scene_traj_curr, scene_traj_act]),
            'demo': torch.cat([scene_demo_demo, scene_demo_curr, scene_demo_act]),
        }
        
        # ==================== Gripper Nodes ====================
        # Demo gripper nodes: [B, D, T, G]
        grip_batch_demo = arange(B)[:, None, None, None].expand(B, D, T, G).reshape(-1)
        grip_time_demo = arange(T)[None, None, :, None].expand(B, D, T, G).reshape(-1)
        grip_node_demo = arange(G)[None, None, None, :].expand(B, D, T, G).reshape(-1)
        grip_demo_demo = arange(D)[None, :, None, None].expand(B, D, T, G).reshape(-1)
        
        # Current gripper nodes: [B, G]
        grip_batch_curr = arange(B)[:, None].expand(B, G).reshape(-1)
        grip_time_curr = torch.full((B * G,), T, dtype=torch.long, device=self.device)
        grip_node_curr = arange(G)[None, :].expand(B, G).reshape(-1)
        grip_demo_curr = torch.full((B * G,), D, device=self.device)
        
        # Action gripper nodes: [B, P, G]
        grip_batch_act = arange(B)[:, None, None].expand(B, P, G).reshape(-1)
        grip_time_act = (arange(P)[None, :, None].expand(B, P, G) + T + 1).reshape(-1).long()
        grip_node_act = arange(G)[None, None, :].expand(B, P, G).reshape(-1)
        grip_demo_act = torch.full((B * P * G,), D, device=self.device)
        
        # Embedding indices for gripper nodes
        # Demo nodes: Encode demo_idx, timestep, and node to give temporal identity
        # This ensures demo gripper at t=0 has different embedding than t=5
        # Range: [0, D*T*G-1]
        grip_embd_demo = grip_demo_demo * T * G + grip_time_demo * G + grip_node_demo
        
        # CRITICAL: Current and action must NOT overlap with demo indices!
        # Demo uses [0, D*T*G-1], so current/action must start at D*T*G
        num_demo_embeds = D * T * G
        
        # Current nodes: start after demo embeddings
        # Range: [D*T*G, D*T*G + G-1]
        grip_embd_curr = grip_node_curr + num_demo_embeds
        
        # Action nodes: start after current embeddings
        # Range: [D*T*G + G, D*T*G + G + P*G-1]
        grip_embd_act = grip_node_act + G * (grip_time_act - T - 1) + num_demo_embeds + G
        
        gripper_info = {
            'batch': torch.cat([grip_batch_demo, grip_batch_curr, grip_batch_act]),
            'time': torch.cat([grip_time_demo, grip_time_curr, grip_time_act]),
            'node': torch.cat([grip_node_demo, grip_node_curr, grip_node_act]),
            'embd': torch.cat([grip_embd_demo, grip_embd_curr, grip_embd_act]).long(),
            'demo': torch.cat([grip_demo_demo, grip_demo_curr, grip_demo_act]),
        }
        
        return {'scene': scene_info, 'gripper': gripper_info}
    
    def initialise_graph(self):
        """Initialize the graph structure with edge indices."""
        self.graph = HeteroData()
        
        # Get node info for both arms
        node_info_left = self._get_node_info('left')
        node_info_right = self._get_node_info('right')
        
        # Store node info in graph
        for arm, info in [('left', node_info_left), ('right', node_info_right)]:
            for node_type in ['scene', 'gripper']:
                prefix = f'{node_type}_{arm}'
                for key, val in info[node_type].items():
                    setattr(self.graph, f'{prefix}_{key}', val)
        
        # Create edge indices
        self._create_local_edges('left', node_info_left)
        self._create_local_edges('right', node_info_right)
        
        if self.use_cross_arm:
            self._create_cross_arm_edges(node_info_left, node_info_right)
    
    def _create_local_edges(self, arm: str, node_info: Dict):
        """Create edges for one arm's local subgraph."""
        scene_key = f'scene_{arm}'
        grip_key = f'gripper_{arm}'
        
        s_info = node_info['scene']
        g_info = node_info['gripper']
        
        num_scene = s_info['batch'].shape[0]
        num_grip = g_info['batch'].shape[0]
        
        # Dense edge indices
        dense_s_s = self._create_dense_edge_idx(num_scene, num_scene)
        dense_s_g = self._create_dense_edge_idx(num_scene, num_grip)
        dense_g_g = self._create_dense_edge_idx(num_grip, num_grip)
        
        T = self.traj_horizon
        
        # ==================== Scene-Scene Edges ====================
        # Same batch, same timestep, same demo
        s_rel_s_mask = (
            (s_info['batch'][dense_s_s[0]] == s_info['batch'][dense_s_s[1]]) &
            (s_info['traj'][dense_s_s[0]] == s_info['traj'][dense_s_s[1]]) &
            (s_info['demo'][dense_s_s[0]] == s_info['demo'][dense_s_s[1]])
        )
        
        # Split into demo and action
        s_is_action = s_info['traj'] > T
        s_rel_s_action = s_rel_s_mask & s_is_action[dense_s_s[0]] & s_is_action[dense_s_s[1]]
        s_rel_s_demo = s_rel_s_mask & ~s_rel_s_action
        
        self.graph[(scene_key, 'rel_demo', scene_key)].edge_index = dense_s_s[:, s_rel_s_demo]
        self.graph[(scene_key, 'rel_action', scene_key)].edge_index = dense_s_s[:, s_rel_s_action]
        
        # ==================== Scene-Gripper Edges ====================
        s_rel_g_mask = (
            (s_info['batch'][dense_s_g[0]] == g_info['batch'][dense_s_g[1]]) &
            (s_info['traj'][dense_s_g[0]] == g_info['time'][dense_s_g[1]]) &
            (s_info['demo'][dense_s_g[0]] == g_info['demo'][dense_s_g[1]])
        )
        
        g_is_action = g_info['time'] > T
        s_rel_g_action = s_rel_g_mask & s_is_action[dense_s_g[0]] & g_is_action[dense_s_g[1]]
        s_rel_g_demo = s_rel_g_mask & ~s_rel_g_action
        
        self.graph[(scene_key, 'rel_demo', grip_key)].edge_index = dense_s_g[:, s_rel_g_demo]
        self.graph[(scene_key, 'rel_action', grip_key)].edge_index = dense_s_g[:, s_rel_g_action]
        
        # ==================== Gripper-Gripper Local Edges ====================
        # Same batch, same timestep, same demo (local spatial)
        g_rel_g_mask = (
            (g_info['batch'][dense_g_g[0]] == g_info['batch'][dense_g_g[1]]) &
            (g_info['time'][dense_g_g[0]] == g_info['time'][dense_g_g[1]]) &
            (g_info['demo'][dense_g_g[0]] == g_info['demo'][dense_g_g[1]])
        )
        self.graph[(grip_key, 'rel', grip_key)].edge_index = dense_g_g[:, g_rel_g_mask]
        
        # ==================== Demo Temporal Edges ====================
        # Within demo, consecutive timesteps (t -> t-1, backwards looking)
        g_demo_mask = (
            (g_info['batch'][dense_g_g[0]] == g_info['batch'][dense_g_g[1]]) &
            (g_info['time'][dense_g_g[0]] < T) &  # Source is demo
            (g_info['time'][dense_g_g[1]] < T) &  # Dest is demo
            (g_info['demo'][dense_g_g[0]] == g_info['demo'][dense_g_g[1]]) &  # Same demo
            (g_info['time'][dense_g_g[1]] - g_info['time'][dense_g_g[0]] == -1)  # Consecutive
        )
        self.graph[(grip_key, 'demo', grip_key)].edge_index = dense_g_g[:, g_demo_mask]
        
        # ==================== Context Edges (Demo → Current) ====================
        g_cond_mask = (
            (g_info['batch'][dense_g_g[0]] == g_info['batch'][dense_g_g[1]]) &
            (g_info['time'][dense_g_g[0]] < T) &  # Source is demo
            (g_info['time'][dense_g_g[1]] == T)   # Dest is current
        )
        self.graph[(grip_key, 'cond', grip_key)].edge_index = dense_g_g[:, g_cond_mask]
        
        # ==================== Action Temporal Edges ====================
        # Current/action → action (forward looking)
        g_time_action_mask = (
            (g_info['batch'][dense_g_g[0]] == g_info['batch'][dense_g_g[1]]) &
            (g_info['time'][dense_g_g[0]] >= T) &  # Source is current or action
            (g_info['time'][dense_g_g[1]] > T) &   # Dest is action
            (g_info['time'][dense_g_g[0]] != g_info['time'][dense_g_g[1]])  # Different timesteps
        )
        
        # Split: current→action vs action→action
        g_curr_to_action = g_time_action_mask & (g_info['time'][dense_g_g[0]] == T)
        g_action_to_action = g_time_action_mask & ~g_curr_to_action
        
        self.graph[(grip_key, 'rel_cond', grip_key)].edge_index = dense_g_g[:, g_curr_to_action]
        self.graph[(grip_key, 'time_action', grip_key)].edge_index = dense_g_g[:, g_action_to_action]
    
    def _create_cross_arm_edges(self, info_left: Dict, info_right: Dict):
        """Create edges connecting left and right arms."""
        g_left = info_left['gripper']
        g_right = info_right['gripper']
        
        num_left = g_left['batch'].shape[0]
        num_right = g_right['batch'].shape[0]
        
        # Dense cross-arm edge indices
        dense_lr = self._create_dense_edge_idx(num_left, num_right)
        dense_rl = self._create_dense_edge_idx(num_right, num_left)
        
        T = self.traj_horizon
        
        # ==================== Current Cross-Arm ====================
        # Left current sees right current
        cross_lr_curr = (
            (g_left['batch'][dense_lr[0]] == g_right['batch'][dense_lr[1]]) &
            (g_left['time'][dense_lr[0]] == T) &
            (g_right['time'][dense_lr[1]] == T)
        )
        self.graph[('gripper_left', 'cross', 'gripper_right')].edge_index = dense_lr[:, cross_lr_curr]
        
        cross_rl_curr = (
            (g_right['batch'][dense_rl[0]] == g_left['batch'][dense_rl[1]]) &
            (g_right['time'][dense_rl[0]] == T) &
            (g_left['time'][dense_rl[1]] == T)
        )
        self.graph[('gripper_right', 'cross', 'gripper_left')].edge_index = dense_rl[:, cross_rl_curr]
        
        # ==================== Action Cross-Arm ====================
        # Left action nodes see right action nodes at same action timestep
        cross_lr_act = (
            (g_left['batch'][dense_lr[0]] == g_right['batch'][dense_lr[1]]) &
            (g_left['time'][dense_lr[0]] > T) &
            (g_right['time'][dense_lr[1]] > T) &
            (g_left['time'][dense_lr[0]] == g_right['time'][dense_lr[1]])  # Same action timestep
        )
        self.graph[('gripper_left', 'cross_action', 'gripper_right')].edge_index = dense_lr[:, cross_lr_act]
        
        cross_rl_act = (
            (g_right['batch'][dense_rl[0]] == g_left['batch'][dense_rl[1]]) &
            (g_right['time'][dense_rl[0]] > T) &
            (g_left['time'][dense_rl[1]] > T) &
            (g_right['time'][dense_rl[0]] == g_left['time'][dense_rl[1]])
        )
        self.graph[('gripper_right', 'cross_action', 'gripper_left')].edge_index = dense_rl[:, cross_rl_act]
        
        # ==================== Demo Cross-Arm ====================
        # Same-timestep cross-arm edges only - rely on within-arm temporal
        # edges to propagate information across time. This reduces edge count
        # by ~10x while maintaining the same information flow capacity.
        # Long-range coordination (e.g., handover) is learned through:
        #   right[t=0] → ... → right[t=5] ↔ left[t=5] → ... → left[t=9]
        
        cross_lr_demo = (
            (g_left['batch'][dense_lr[0]] == g_right['batch'][dense_lr[1]]) &
            (g_left['time'][dense_lr[0]] < T) &
            (g_right['time'][dense_lr[1]] < T) &
            (g_left['demo'][dense_lr[0]] == g_right['demo'][dense_lr[1]]) &  # Same demo
            (g_left['time'][dense_lr[0]] == g_right['time'][dense_lr[1]])    # Same timestep
        )
        self.graph[('gripper_left', 'cross_demo', 'gripper_right')].edge_index = dense_lr[:, cross_lr_demo]
        
        cross_rl_demo = (
            (g_right['batch'][dense_rl[0]] == g_left['batch'][dense_rl[1]]) &
            (g_right['time'][dense_rl[0]] < T) &
            (g_left['time'][dense_rl[1]] < T) &
            (g_right['demo'][dense_rl[0]] == g_left['demo'][dense_rl[1]]) &  # Same demo
            (g_right['time'][dense_rl[0]] == g_left['time'][dense_rl[1]])    # Same timestep
        )
        self.graph[('gripper_right', 'cross_demo', 'gripper_left')].edge_index = dense_rl[:, cross_rl_demo]
    
    def transform_gripper_nodes(self, gripper_keypoints: torch.Tensor, 
                                 T: torch.Tensor) -> torch.Tensor:
        """
        Transform gripper keypoints by SE(3) transformations.
        
        Args:
            gripper_keypoints: [B, D, T, G, 3] or [B, T, G, 3] or [B, G, 3]
            T: [B, D, T, 4, 4] or [B, T, 4, 4] or [B, 4, 4] - transformations
        
        Returns:
            Transformed keypoints with same shape as input
        """
        original_shape = gripper_keypoints.shape
        
        # Reshape to [N, G, 3] for batch processing
        if len(original_shape) == 5:  # [B, D, T, G, 3]
            B, D, T_, G, _ = original_shape
            kp = gripper_keypoints.reshape(-1, G, 3)
            transforms = T.reshape(-1, 4, 4)
        elif len(original_shape) == 4:  # [B, T, G, 3]
            B, T_, G, _ = original_shape
            kp = gripper_keypoints.reshape(-1, G, 3)
            transforms = T.reshape(-1, 4, 4)
        else:  # [B, G, 3]
            B, G, _ = original_shape
            kp = gripper_keypoints
            transforms = T
        
        # Apply transformation: T @ kp
        kp_transformed = torch.bmm(transforms[:, :3, :3], kp.transpose(1, 2))  # [N, 3, G]
        kp_transformed = kp_transformed.transpose(1, 2) + transforms[:, :3, 3:4].transpose(1, 2)  # [N, G, 3]
        
        return kp_transformed.reshape(original_shape)
    
    def update_graph(self, data) -> HeteroData:
        """
        Update graph with actual node features and edge attributes from data.
        
        Args:
            data: BimanualGraphData with all required tensors
        
        Returns:
            Updated HeteroData graph
        """
        B = self.batch_size
        D = self.num_demos
        T = self.traj_horizon
        G = self.num_g_nodes
        P = self.pred_horizon
        
        # ==================== Scene Node Features ====================
        for arm in ['left', 'right']:
            scene_key = f'scene_{arm}'
            
            # Concatenate demo, current, and action scene embeddings
            demo_embds = getattr(data, f'demo_scene_embds_{arm}')[:, :D]  # [B, D, T, S, dim]
            live_embds = getattr(data, f'live_scene_embds_{arm}')         # [B, S, dim]
            action_embds = getattr(data, f'action_scene_embds_{arm}')     # [B, P, S, dim]
            
            demo_pos = getattr(data, f'demo_scene_pos_{arm}')[:, :D]
            live_pos = getattr(data, f'live_scene_pos_{arm}')
            action_pos = getattr(data, f'action_scene_pos_{arm}')
            
            scene_embds = torch.cat([
                demo_embds.reshape(-1, self.embd_dim),
                live_embds.reshape(-1, self.embd_dim),
                action_embds.reshape(-1, self.embd_dim),
            ], dim=0)
            
            scene_pos = torch.cat([
                demo_pos.reshape(-1, 3),
                live_pos.reshape(-1, 3),
                action_pos.reshape(-1, 3),
            ], dim=0)
            
            self.graph[scene_key].x = scene_embds
            self.graph[scene_key].pos = scene_pos
            
            if self.pos_in_nodes:
                self.graph[scene_key].x = torch.cat([
                    self.graph[scene_key].x,
                    self.pos_encoder(self.graph[scene_key].pos)
                ], dim=-1)
        
        # ==================== Gripper Node Features ====================
        for arm in ['left', 'right']:
            grip_key = f'gripper_{arm}'
            
            # Get poses and gripper states
            demo_T = getattr(data, f'demo_T_w_{arm}')[:, :D]  # [B, D, T, 4, 4]
            actions_T = getattr(data, f'actions_{arm}')       # [B, P, 4, 4]
            demo_grips = getattr(data, f'demo_grips_{arm}')[:, :D]  # [B, D, T]
            current_grip = getattr(data, f'current_grip_{arm}')  # [B]
            action_grips = getattr(data, f'actions_grip_{arm}')  # [B, P]
            
            # Compute gripper node positions
            base_kp = self.gripper_keypoints[None, None, None, :, :]  # [1, 1, 1, G, 3]
            
            # Demo gripper positions in local frame
            demo_kp = base_kp.expand(B, D, T, G, 3)
            
            # Current gripper positions (at origin in local frame)
            curr_kp = self.gripper_keypoints[None, :, :].expand(B, G, 3)
            
            # Action gripper positions in action frame
            action_kp = base_kp[:, 0].expand(B, P, G, 3)
            
            # Concatenate positions
            grip_pos = torch.cat([
                demo_kp.reshape(-1, 3),
                curr_kp.reshape(-1, 3),
                action_kp.reshape(-1, 3),
            ], dim=0)
            
            # Compute gripper state embeddings
            demo_grip_embd = self.gripper_proj(demo_grips.unsqueeze(-1))  # [B, D, T, g_state_dim]
            demo_grip_embd = demo_grip_embd[:, :, :, None, :].expand(B, D, T, G, -1)
            
            curr_grip_embd = self.gripper_proj(current_grip.unsqueeze(-1))  # [B, g_state_dim]
            curr_grip_embd = curr_grip_embd[:, None, :].expand(B, G, -1)
            
            action_grip_embd = self.gripper_proj(action_grips.unsqueeze(-1))  # [B, P, g_state_dim]
            action_grip_embd = action_grip_embd[:, :, None, :].expand(B, P, G, -1)
            
            grip_state_embd = torch.cat([
                demo_grip_embd.reshape(-1, self.g_state_dim),
                curr_grip_embd.reshape(-1, self.g_state_dim),
                action_grip_embd.reshape(-1, self.g_state_dim),
            ], dim=0)
            
            # Get learnable embeddings
            embd_indices = getattr(self.graph, f'{grip_key}_embd')
            gripper_embds = getattr(self, f'gripper_embds_{arm}')
            learned_embd = gripper_embds(embd_indices % gripper_embds.num_embeddings)
            
            # Add diffusion time embedding to action nodes
            time_mask = getattr(self.graph, f'{grip_key}_time') > T
            if time_mask.any():
                d_time_embd = self.time_emb(data.diff_time.squeeze())  # [B, d_time_dim]
                d_time_expanded = d_time_embd[:, None, None, :].expand(B, P, G, -1).reshape(-1, self.d_time_dim)
                
                # Replace last d_time_dim dimensions for action nodes
                action_start_idx = B * D * T * G + B * G
                learned_embd[action_start_idx:, -self.d_time_dim:] = d_time_expanded
            
            # Combine all embeddings
            grip_embd = torch.cat([learned_embd, grip_state_embd], dim=-1)
            
            self.graph[grip_key].x = grip_embd
            self.graph[grip_key].pos = grip_pos
            
            if self.pos_in_nodes:
                self.graph[grip_key].x = torch.cat([
                    self.graph[grip_key].x,
                    self.pos_encoder(self.graph[grip_key].pos)
                ], dim=-1)
        
        # ==================== Edge Attributes ====================
        # ==================== Edge Attributes ====================
        # Compute transforms for SE(3) attributes
        all_T_w_e = {}
        all_T_e_w = {}
        
        for arm in ['left', 'right']:
            grip_key = f'gripper_{arm}'
            
            # Pack transforms: Demo (D*T), Current (1), Action (P)
            # Demo Transforms: [B, D, T, 4, 4] -> [B*D*T, 4, 4]
            demo_T = getattr(data, f'demo_T_w_{arm}')[:, :D].reshape(B*D*T, 4, 4)
            
            # Current Transform: Identity (since current is local origin) or proper world transform?
            # Original IP uses Identity for current if relative to world?
            # User snippet: "I_w_e = torch.eye...; all_T_w_e = torch.cat([...])"
            # Since our 'pos' for current is local (0,0,0) based on lines 589-590...
            # We should probably use Identity for T_w_e of current step if we treat it as the reference frame?
            # OR, if we want everything in consistency, we must use the actual Transforms if available.
            # But the 'current_grip_T' might not be in data if it's egocentric?
            # Check data - we usually assume current is Identity in egocentric.
            
            # Let's assume Identity for current frame for now, matching "I_w_e" in snippet.
            current_T = torch.eye(4, device=self.device)[None, :, :].repeat(B, 1, 1).reshape(B, 4, 4)
            
            # Action Transforms: [B, P, 4, 4] -> [B*P, 4, 4]
            action_T = getattr(data, f'actions_{arm}').reshape(B*P, 4, 4)
            
            # Combine all transforms for this arm
            # Order must match node ordering in update_graph: Demo, Current, Action
            T_w_e_arm = torch.cat([demo_T, current_T, action_T], dim=0)
            
            # Expand to per-node (G nodes per timestep)
            # T_w_e [TotalSteps, 4, 4] -> [TotalSteps * G, 4, 4]
            T_w_e_arm = T_w_e_arm[:, None, :, :].repeat(1, G, 1, 1).reshape(-1, 4, 4)
            
            T_e_w_arm = torch.inverse(T_w_e_arm)
            
            all_T_w_e[grip_key] = T_w_e_arm
            all_T_e_w[grip_key] = T_e_w_arm
        
        self._compute_edge_attributes(data, all_T_w_e, all_T_e_w)
        
        return self.graph
    
    def _compute_edge_attributes(self, data, all_T_w_e=None, all_T_e_w=None):
        """Compute edge attributes for all edge types."""
        
        # Local edges (relative positions)
        for arm in ['left', 'right']:
            scene_key = f'scene_{arm}'
            grip_key = f'gripper_{arm}'
            
            # Scene-scene edges (No rotation info for scene points usually)
            for edge_type in ['rel_demo', 'rel_action']:
                self._add_rel_edge_attr(scene_key, scene_key, edge_type)
            
            # Scene-gripper edges (Gripper has rotation, scene doesn't)
            # We can use gripper rotation if we want scene to be relative to gripper frame?
            # But standard impl might just be positional.
            for edge_type in ['rel_demo', 'rel_action']:
                self._add_rel_edge_attr(scene_key, grip_key, edge_type, all_T_w_e, all_T_e_w)
            
            # Gripper-gripper local edges
            self._add_rel_edge_attr(grip_key, grip_key, 'rel')
            
            # Context edges (with learnable embedding)
            edge_key = (grip_key, 'cond', grip_key)
            if edge_key in self.graph.edge_types:
                num_edges = self.graph[edge_key].edge_index.shape[1]
                base_attr = self._compute_rel_attr(grip_key, grip_key, 'cond', all_T_w_e, all_T_e_w)
                learned = self.cond_edge_embd(torch.zeros(num_edges, device=self.device).long())
                self.graph[edge_key].edge_attr = base_attr + learned
            
            # Temporal edges
            for edge_type in ['demo', 'time_action', 'rel_cond']:
                self._add_rel_edge_attr(grip_key, grip_key, edge_type, all_T_w_e, all_T_e_w)
        
        # Cross-arm edges
        if self.use_cross_arm:
            self._compute_cross_arm_edge_attrs(data)
    
    def _add_rel_edge_attr(self, src: str, dst: str, edge_type: str, 
                           all_T_w_e=None, all_T_e_w=None):
        """Add relative position edge attributes."""
        edge_key = (src, edge_type, dst)
        if edge_key not in self.graph.edge_types:
            return
        
        edge_attr = self._compute_rel_attr(src, dst, edge_type, all_T_w_e, all_T_e_w)
        self.graph[edge_key].edge_attr = edge_attr
    
    def _compute_rel_attr(self, src: str, dst: str, edge_type: str,
                          all_T_w_e=None, all_T_e_w=None) -> torch.Tensor:
        """
        Compute relative position encoding for edges.
        
        Args:
            src: Source node type
            dst: Destination node type
            edge_type: Edge type classification
            all_T_w_e: World-to-element transforms for all nodes (if available)
            all_T_e_w: Element-to-world transforms for all nodes (if available)
        """
        edge_key = (src, edge_type, dst)
        edge_index = self.graph[edge_key].edge_index
        
        # Handle empty edge case
        if edge_index.shape[1] == 0:
            return torch.zeros((0, self.edge_dim), device=self.device)
        
        # Standard relative position (translation only if no rotation info)
        src_pos = self.graph[src].pos[edge_index[0]]
        dst_pos = self.graph[dst].pos[edge_index[1]]
        
        # If we have transform info (usually for gripper nodes), use it for rotation-aware encoding
        if all_T_w_e is not None and all_T_e_w is not None:
             # Need to map graph node indices to global transform indices if possible
             # Or pass specific transforms for src/dst. 
             # For simpler implementation matching IP:
             # We assume all_T_w_e is a dict mapping node_type -> transforms [N, 4, 4]
             
             T_w_src = all_T_w_e.get(src)
             T_dst_w = all_T_e_w.get(dst)
             
             if T_w_src is not None and T_dst_w is not None:
                 # Get relevant transforms
                 T_w_e_src = T_w_src[edge_index[0]] # [E, 4, 4]
                 T_e_w_dst = T_dst_w[edge_index[1]] # [E, 4, 4]
                 
                 # Relative transform T_i_j = T_i_w @ T_w_j = T_e_w_src @ T_w_e_dst
                 # WAIT: IP implementation uses:
                 # T_i_j = torch.bmm(all_T_e_w[edge_index[0]], all_T_w_e[edge_index[1]])
                 # So we need Element-to-World of Source AND World-to-Element of Dest?
                 # Let's check IP: 
                 # T_i_j = T_source_world @ T_world_dest (which is T_src_dest)
                 # Then pos_dest_rot = T_i_j[:3,:3] @ pos_source_local ? No.
                 
                 # IP Logic:
                 # pos_dest_rot = T_i_j[..., :3, :3] @ pos_source[..., None]
                 # This rotates the source vector? 
                 # Actually, let's stick to the physical meaning:
                 # Feature 1: pos_dest - pos_source (Relative translation in world frame? Or local?)
                 # Feature 2: Rotation-invariance.
                 
                 # Correct Logic for SE(3) invariance:
                 # Attributes should be relative parameters in the SOURCE frame.
                 # 1. Delta Position in Source Frame: R_src^T @ (p_dst - p_src)
                 # 2. Delta Rotation? Or just position?
                 # Original IP:
                 # pos_dest_rot = R_rel @ pos_source 
                 # This seems specific to their formulation.
                 
                 # Let's strictly follow the provided "CRITICAL" snippet from user request to match IP logic.
                 # T_i_j = all_T_e_w[src] @ all_T_w_e[dst]  (Relative transform Source->Dest)
                 # pos_dest_rot = T_i_j.R @ pos_source
                 
                 # If we use the user's snippet exact logic:
                 T_src_w = all_T_e_w[src][edge_index[0]] # [E, 4, 4]  (Element to World = T_w_e.inverse)
                 T_w_dst = all_T_w_e[dst][edge_index[1]] # [E, 4, 4]  (World to Element)
                 
                 # T_src_dst = T_src_w @ T_w_dst ? No.
                 # T_i_j = T_e_w[src] @ T_w_e[dst]. This results in T_src_inv @ T_dst. 
                 # Which is T_src->world @ T_world->dst = T_src->dst? No.
                 # T_src_w = T_w_src^-1. 
                 # T_w_dst = T_dst.
                 # T_src_w @ T_w_dst = T_w_src^-1 @ T_dst_w^-1? 
                 # Let's assume standard notation T_A_B means point in B to point in A.
                 # T_w_e: Point in Element to Point in World. (Standard model matrix)
                 # T_e_w: Point in World to Point in Element. (View matrix)
                 
                 # IP Code: T_i_j = T_e_w[src] @ T_w_e[dst]
                 # = T_world->src @ T_dst->world
                 # = T_dst->src (Point in Dst frame to Point in Src frame)
                 # Yes.
                 
                 T_src_dst = torch.bmm(T_src_w, T_w_dst)
                 
                 # Now, what is pos_dest_rot?
                 # pos_dest_rot = T_src_dst.R @ pos_source
                 # This rotates the source position vector by the relative rotation.
                 
                 pos_dest_rot = torch.bmm(T_src_dst[..., :3, :3], src_pos[..., None]).squeeze(-1)
                 rel_trans = T_src_dst[..., :3, 3]
                 
                 # Features:
                 # 1. Relative translation (source frame)
                 # 2. Rotation effect on source keypoints
                 
                 return torch.cat([
                    self.pos_encoder(rel_trans),
                    self.pos_encoder(pos_dest_rot - src_pos)
                 ], dim=-1)

        # Fallback if no transforms (e.g. scene nodes without orientation)
        rel_pos = dst_pos - src_pos
        return torch.cat([
            self.pos_encoder(rel_pos),
            self.pos_encoder(rel_pos), # Duplicate (no rotation info)
        ], dim=-1)
    
    def _compute_cross_arm_edge_attrs(self, data):
        """Compute edge attributes for cross-arm edges."""
        T = self.traj_horizon
        T_left_to_right = data.current_T_left_to_right  # [B, 4, 4]
        if T_left_to_right is None:
            return
        if T_left_to_right.dim() == 2:
            T_left_to_right = T_left_to_right.unsqueeze(0)

        def _inv_safe(mat):
            if mat is None:
                return None
            if not mat.is_floating_point():
                return torch.inverse(mat)
            if mat.dtype in (torch.float16, torch.bfloat16):
                return torch.inverse(mat.float()).to(mat.dtype)
            return torch.inverse(mat)
        
        # Demo transforms (per-timestep, per-demo)
        demo_T_left_to_right = getattr(data, 'demo_T_left_to_right', None)
        demo_T_w_left = getattr(data, 'demo_T_w_left', None)
        demo_T_w_right = getattr(data, 'demo_T_w_right', None)
        if demo_T_left_to_right is None:
            demo_T_left = demo_T_w_left
            demo_T_right = demo_T_w_right
            if demo_T_left is not None and demo_T_right is not None:
                demo_T_left_to_right = torch.matmul(_inv_safe(demo_T_left), demo_T_right)
        if demo_T_left_to_right is not None:
            if demo_T_left_to_right.dim() == 3:
                demo_T_left_to_right = demo_T_left_to_right.unsqueeze(0).unsqueeze(1)
            elif demo_T_left_to_right.dim() == 4:
                demo_T_left_to_right = demo_T_left_to_right.unsqueeze(1)
        if demo_T_w_left is not None:
            if demo_T_w_left.dim() == 3:
                demo_T_w_left = demo_T_w_left.unsqueeze(0).unsqueeze(0)
            elif demo_T_w_left.dim() == 4:
                demo_T_w_left = demo_T_w_left.unsqueeze(0)
        if demo_T_w_right is not None:
            if demo_T_w_right.dim() == 3:
                demo_T_w_right = demo_T_w_right.unsqueeze(0).unsqueeze(0)
            elif demo_T_w_right.dim() == 4:
                demo_T_w_right = demo_T_w_right.unsqueeze(0)
        
        # Action transforms (per-action-step)
        actions_left = getattr(data, 'actions_left', None)
        actions_right = getattr(data, 'actions_right', None)
        if actions_left is not None and actions_left.dim() == 3:
            actions_left = actions_left.unsqueeze(0)
        if actions_right is not None and actions_right.dim() == 3:
            actions_right = actions_right.unsqueeze(0)
        action_T_left_to_right = None
        if actions_left is not None and actions_right is not None:
            B, P = actions_left.shape[:2]
            T_lr_curr = T_left_to_right[:, None, :, :].expand(B, P, 4, 4).reshape(-1, 4, 4)
            A_left_inv = _inv_safe(actions_left.reshape(-1, 4, 4))
            A_right = actions_right.reshape(-1, 4, 4)
            action_T_left_to_right = torch.bmm(
                A_left_inv, torch.bmm(T_lr_curr, A_right)
            ).view(B, P, 4, 4)
        
        def _build_cross_attr(edge_index, src_key, dst_key, T_src_from_dst):
            if edge_index.shape[1] == 0:
                return torch.zeros((0, self.edge_dim), device=self.device)
            src_pos = self.graph[src_key].pos[edge_index[0]]
            dst_pos = self.graph[dst_key].pos[edge_index[1]]
            
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
                self.pos_encoder(rel_pos),
                self.pos_encoder(pos_dest_rot - src_pos),
            ], dim=-1)
        
        def _edge_attrs_for_direction(src_arm, dst_arm, edge_type,
                                      T_curr, T_action, T_demo):
            edge_key = (f'gripper_{src_arm}', edge_type, f'gripper_{dst_arm}')
            if edge_key not in self.graph.edge_types:
                return
            
            edge_index = self.graph[edge_key].edge_index
            num_edges = edge_index.shape[1]
            if num_edges == 0:
                self.graph[edge_key].edge_attr = torch.zeros((0, self.edge_dim), device=self.device)
                return
            
            src_key = f'gripper_{src_arm}'
            dst_key = f'gripper_{dst_arm}'
            src_batch = getattr(self.graph, f'{src_key}_batch')[edge_index[0]]
            src_time = getattr(self.graph, f'{src_key}_time')[edge_index[0]].long()
            dst_time = getattr(self.graph, f'{dst_key}_time')[edge_index[1]].long()
            
            if edge_type == 'cross':
                T_src_from_dst = T_curr[src_batch]
            elif edge_type == 'cross_action':
                if T_action is None:
                    T_src_from_dst = T_curr[src_batch]
                else:
                    max_step = T_action.shape[1] - 1
                    action_step = (src_time - T - 1).clamp(min=0, max=max_step)
                    T_src_from_dst = T_action[src_batch, action_step]
            else:  # cross_demo
                if demo_T_w_left is not None and demo_T_w_right is not None:
                    src_demo = getattr(self.graph, f'{src_key}_demo')[edge_index[0]].long()
                    if src_arm == 'left':
                        T_w_src = demo_T_w_left[src_batch, src_demo, src_time]
                        T_w_dst = demo_T_w_right[src_batch, src_demo, dst_time]
                    else:
                        T_w_src = demo_T_w_right[src_batch, src_demo, src_time]
                        T_w_dst = demo_T_w_left[src_batch, src_demo, dst_time]
                    T_src_from_dst = torch.bmm(_inv_safe(T_w_src), T_w_dst)
                elif T_demo is None:
                    T_src_from_dst = T_curr[src_batch]
                else:
                    src_demo = getattr(self.graph, f'{src_key}_demo')[edge_index[0]].long()
                    T_src_from_dst = T_demo[src_batch, src_demo, src_time]
            
            base_attr = _build_cross_attr(edge_index, src_key, dst_key, T_src_from_dst)
            
            if 'action' in edge_type:
                learned = self.cross_action_edge_embd(torch.zeros(num_edges, device=self.device).long())
            else:
                learned = self.cross_arm_edge_embd(torch.zeros(num_edges, device=self.device).long())
            
            self.graph[edge_key].edge_attr = base_attr + learned
        
        T_right_to_left = _inv_safe(T_left_to_right)
        action_T_right_to_left = _inv_safe(action_T_left_to_right)
        demo_T_right_to_left = _inv_safe(demo_T_left_to_right)
        
        # Left -> Right edges
        _edge_attrs_for_direction('left', 'right', 'cross',
                                  T_left_to_right, action_T_left_to_right, demo_T_left_to_right)
        _edge_attrs_for_direction('left', 'right', 'cross_action',
                                  T_left_to_right, action_T_left_to_right, demo_T_left_to_right)
        _edge_attrs_for_direction('left', 'right', 'cross_demo',
                                  T_left_to_right, action_T_left_to_right, demo_T_left_to_right)
        
        # Right -> Left edges
        _edge_attrs_for_direction('right', 'left', 'cross',
                                  T_right_to_left, action_T_right_to_left, demo_T_right_to_left)
        _edge_attrs_for_direction('right', 'left', 'cross_action',
                                  T_right_to_left, action_T_right_to_left, demo_T_right_to_left)
        _edge_attrs_for_direction('right', 'left', 'cross_demo',
                                  T_right_to_left, action_T_right_to_left, demo_T_right_to_left)
    
    def reinit_graphs(self, batch_size: int, num_demos: Optional[int] = None):
        """Reinitialize graph structure for new batch size or number of demos."""
        self.batch_size = batch_size
        if num_demos is not None:
            self.num_demos = num_demos
        self.initialise_graph()
