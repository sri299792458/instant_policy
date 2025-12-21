"""
Bimanual AGI Model for dual-arm manipulation.

Extends the single-arm AGI model to handle two end-effectors
with proper spatial invariance and coordination.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MLP
from typing import Dict, Optional

from ip.models.scene_encoder import SceneEncoder
from ip.models.graph_transformer import GraphTransformer
from ip.utils.common_utils import dfs_freeze
from bimanual_graph_rep import BimanualGraphRep


class BimanualAGI(nn.Module):
    """
    Bimanual Action Generation through In-context learning.
    
    Extends AGI to handle dual-arm manipulation with:
    - Dual egocentric scene encoding
    - Cross-arm coordination through graph edges
    - Separate prediction heads for each arm
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_demos = config['num_demos']
        self.num_demos_in_use = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.local_embd_dim = config['local_nn_dim']
        self.batch_size = config['batch_size']
        self.num_scene_nodes = config['num_scenes_nodes']
        self.pred_horizon = config['pre_horizon']
        self.num_layers = config['num_layers']
        self.shared_encoder = config.get('shared_scene_encoder', True)
        
        # ============== Scene Encoders ==============
        # Can be shared or separate for each arm
        self.scene_encoder = SceneEncoder(
            num_freqs=10,
            embd_dim=config['local_nn_dim']
        ).to(config['device'])
        
        if not self.shared_encoder:
            self.scene_encoder_right = SceneEncoder(
                num_freqs=10,
                embd_dim=config['local_nn_dim']
            ).to(config['device'])
        
        # Load pretrained encoder if specified
        if config.get('pre_trained_encoder', False):
            self.scene_encoder.load_state_dict(
                torch.load(config['scene_encoder_path'])
            )
            if config.get('freeze_encoder', True):
                dfs_freeze(self.scene_encoder)
            
            if not self.shared_encoder:
                self.scene_encoder_right.load_state_dict(
                    torch.load(config['scene_encoder_path'])
                )
                if config.get('freeze_encoder', True):
                    dfs_freeze(self.scene_encoder_right)
        
        # ============== Graph Representation ==============
        self.graph = BimanualGraphRep(config)
        self.graph.initialise_graph()
        
        # ============== Graph Transformers (Staged) ==============
        # 1. Local Encoder: Processes local spatial edges (rel)
        # 2. Cond Encoder: Processes demo context edges (demo, cond, rel_demo, cross_demo)
        # 3. Action Encoder: Processes action edges (time_action, rel_cond, rel_action, cross_action)
        
        in_channels = self.local_embd_dim
        if config.get('pos_in_nodes', True):
            in_channels += self.graph.edge_dim // 2
        
        # Define edge subsets for each encoder
        # This matches the original IP design + bimanual cross edges
        
        # Local edges: Spatial relations within same timestep
        local_edges = [
            ('gripper_left', 'rel', 'gripper_left'),
            ('gripper_right', 'rel', 'gripper_right'),
            ('gripper_left', 'cross', 'gripper_right'), # Cross-arm current
            ('gripper_right', 'cross', 'gripper_left'),
        ]
        
        # Context edges: Demo temporal and demo-to-current
        cond_edges = [
            ('gripper_left', 'demo', 'gripper_left'),
            ('gripper_right', 'demo', 'gripper_right'),
            ('gripper_left', 'cond', 'gripper_left'),
            ('gripper_right', 'cond', 'gripper_right'),
            ('scene_left', 'rel_demo', 'gripper_left'),
            ('scene_right', 'rel_demo', 'gripper_right'),
            ('scene_left', 'rel_demo', 'scene_left'),
            ('scene_right', 'rel_demo', 'scene_right'),
        ]
        if self.graph.use_cross_arm:
            cond_edges.extend([
                ('gripper_left', 'cross_demo', 'gripper_right'),
                ('gripper_right', 'cross_demo', 'gripper_left'),
            ])
            
        # Action edges: Future prediction
        action_edges = [
            ('gripper_left', 'time_action', 'gripper_left'),
            ('gripper_right', 'time_action', 'gripper_right'),
            ('gripper_left', 'rel_cond', 'gripper_left'),
            ('gripper_right', 'rel_cond', 'gripper_right'),
            ('scene_left', 'rel_action', 'gripper_left'),
            ('scene_right', 'rel_action', 'gripper_right'),
            ('scene_left', 'rel_action', 'scene_left'),
            ('scene_right', 'rel_action', 'scene_right'),
        ]
        if self.graph.use_cross_arm:
            action_edges.extend([
                ('gripper_left', 'cross_action', 'gripper_right'),
                ('gripper_right', 'cross_action', 'gripper_left'),
            ])

        # Helper to filter metadata for specific edges
        def filter_metadata(edges):
            return (self.graph.node_types, [e for e in self.graph.edge_types if e in edges])

        self.local_encoder = GraphTransformer(
            in_channels=in_channels,
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=filter_metadata(local_edges),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer'
        ).to(config['device'])

        self.cond_encoder = GraphTransformer(
            in_channels=config['hidden_dim'], # Takes output of local
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=filter_metadata(cond_edges),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer'
        ).to(config['device'])

        self.action_encoder = GraphTransformer(
            in_channels=config['hidden_dim'], # Takes output of cond
            hidden_channels=config['hidden_dim'],
            heads=config['hidden_dim'] // 64,
            num_layers=self.num_layers,
            metadata=filter_metadata(action_edges),
            edge_dim=self.graph.edge_dim,
            dropout=0.0,
            norm='layer'
        ).to(config['device'])
        
        # Compile encoders
        if config.get('compile_model', True):
            self.local_encoder = torch.compile(self.local_encoder, mode="reduce-overhead")
            self.cond_encoder = torch.compile(self.cond_encoder, mode="reduce-overhead")
            self.action_encoder = torch.compile(self.action_encoder, mode="reduce-overhead")
        
        # ============== Prediction Heads (per arm) ==============
        # Left arm
        self.pred_head_trans_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        self.pred_head_rot_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        self.pred_head_grip_left = MLP(
            [config['hidden_dim'], self.local_embd_dim, 1],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        
        # Right arm
        self.pred_head_trans_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        self.pred_head_rot_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 3],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        self.pred_head_grip_right = MLP(
            [config['hidden_dim'], self.local_embd_dim, 1],
            act='GELU', plain_last=True, norm='layer_norm'
        )
        
        # Compile if requested
        if config.get('compile_models', False):
            self.compile_models()
    
    def reinit_graphs(self, batch_size: int, num_demos: Optional[int] = None):
        """Reinitialize graph for new batch size or number of demos."""
        self.batch_size = batch_size
        if num_demos is not None:
            self.num_demos = num_demos
            self.graph.num_demos = num_demos
        self.graph.batch_size = batch_size
        self.graph.initialise_graph()
    
    def compile_models(self):
        """Compile models for faster inference."""
        self.scene_encoder.sa1_module.conv = torch.compile(
            self.scene_encoder.sa1_module.conv, mode="reduce-overhead"
        )
        self.scene_encoder.sa2_module.conv = torch.compile(
            self.scene_encoder.sa2_module.conv, mode="reduce-overhead"
        )
        self.local_encoder = torch.compile(self.local_encoder, mode="reduce-overhead")
        self.cond_encoder = torch.compile(self.cond_encoder, mode="reduce-overhead")
        self.action_encoder = torch.compile(self.action_encoder, mode="reduce-overhead")
    
    def get_demo_scene_emb(self, data, arm: str):
        """Get scene embeddings for demonstrations."""
        pos_demos = getattr(data, f'pos_demos_{arm}')
        batch_demos = getattr(data, f'batch_demos_{arm}')
        
        encoder = self.scene_encoder if self.shared_encoder else (
            self.scene_encoder if arm == 'left' else self.scene_encoder_right
        )
        
        embds, pos, batch = encoder(None, pos_demos, batch_demos)
        
        # Reshape to [B, D, T, S, dim]
        embds = to_dense_batch(embds, batch, fill_value=0)[0]
        bs = embds.shape[0] # Dynamic batch size
        embds = embds.view(
            bs, self.num_demos, self.traj_horizon,
            -1, self.local_embd_dim
        )
        
        pos = to_dense_batch(pos, batch, fill_value=0)[0]
        pos = pos.view(
            self.batch_size, self.num_demos, self.traj_horizon, -1, 3
        )
        
        return embds, pos
    
    def get_live_scene_emb(self, data, arm: str):
        """Get scene embeddings for current observation."""
        pos_obs = getattr(data, f'pos_obs_{arm}')
        batch_obs = data.batch_pos_obs
        
        encoder = self.scene_encoder if self.shared_encoder else (
            self.scene_encoder if arm == 'left' else self.scene_encoder_right
        )
        
        embds, pos, batch = encoder(None, pos_obs, batch_obs)
        embds = to_dense_batch(embds, batch, fill_value=0)[0]
        pos = to_dense_batch(pos, batch, fill_value=0)[0]
        
        return embds, pos
    
    def get_action_scene_emb(self, data, arm: str):
        """Get scene embeddings for action frames (scene transformed by predicted actions)."""
        pos_obs = getattr(data, f'pos_obs_{arm}')
        batch_obs = data.batch_pos_obs
        actions = getattr(data, f'actions_{arm}')  # [B, P, 4, 4]
        
        # Transform observation point cloud by each action
        current_obs = to_dense_batch(pos_obs, batch_obs, fill_value=0)[0]  # [B, N, 3]
        current_obs = current_obs[:, None, ...].repeat(1, self.pred_horizon, 1, 1)  # [B, P, N, 3]
        current_obs = current_obs.view(self.batch_size * self.pred_horizon, -1, 3)
        
        actions_flat = actions.view(-1, 4, 4)  # [B*P, 4, 4]
        
        # Transform: R^T @ (p - t)
        current_obs = current_obs - actions_flat[:, :3, 3][:, None, :]
        current_obs = torch.bmm(
            actions_flat[:, :3, :3].transpose(1, 2),
            current_obs.permute(0, 2, 1)
        ).permute(0, 2, 1)
        
        # Encode
        action_batch = torch.arange(
            current_obs.shape[0], device=current_obs.device
        )[:, None].repeat(1, current_obs.shape[1]).view(-1)
        current_obs_flat = current_obs.reshape(-1, 3)
        
        encoder = self.scene_encoder if self.shared_encoder else (
            self.scene_encoder if arm == 'left' else self.scene_encoder_right
        )
        
        embds, pos, batch = encoder(None, current_obs_flat, action_batch)
        embds = to_dense_batch(embds, batch, fill_value=0)[0]
        embds = embds.view(self.batch_size, self.pred_horizon, -1, self.local_embd_dim)
        
        pos = to_dense_batch(pos, batch, fill_value=0)[0]
        pos = pos.view(self.batch_size, self.pred_horizon, -1, 3)
        
        return embds, pos
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: BimanualGraphData with all required tensors
            
        Returns:
            preds_left: [B, P, G, 7] predictions for left arm (trans, rot, grip)
            preds_right: [B, P, G, 7] predictions for right arm
        """
        # ============== Scene Encoding ==============
        # Get embeddings for both arms
        for arm in ['left', 'right']:
            if not hasattr(data, f'demo_scene_embds_{arm}'):
                embds, pos = self.get_demo_scene_emb(data, arm)
                setattr(data, f'demo_scene_embds_{arm}', embds)
                setattr(data, f'demo_scene_pos_{arm}', pos)
            
            if not hasattr(data, f'live_scene_embds_{arm}'):
                embds, pos = self.get_live_scene_emb(data, arm)
                setattr(data, f'live_scene_embds_{arm}', embds)
                setattr(data, f'live_scene_pos_{arm}', pos)
            
            if not hasattr(data, f'action_scene_embds_{arm}'):
                embds, pos = self.get_action_scene_emb(data, arm)
                setattr(data, f'action_scene_embds_{arm}', embds)
                setattr(data, f'action_scene_pos_{arm}', pos)
        
        # ============== Graph Update ==============
        self.graph.update_graph(data)
        
        # ============== Graph Transformer Forward (Staged) ==============
        torch.compiler.cudagraph_mark_step_begin()
        
        # 1. Local Spatial Processing
        # Only processes local edges to understand immediate spatial relations
        x_dict = self.local_encoder(
            self.graph.graph.x_dict,
            self.graph.graph.edge_index_dict,
            self.graph.graph.edge_attr_dict
        )
        
        # 2. Context Propagation
        # Propagates information from demos to current context
        x_dict = self.cond_encoder(
            x_dict,
            self.graph.graph.edge_index_dict, # to_hetero filters automatically? No, we pass all but encoder only uses its subset
            self.graph.graph.edge_attr_dict
        )
        
        # 3. Action Prediction
        # Propagates from context to future actions
        x_dict = self.action_encoder(
            x_dict,
            self.graph.graph.edge_index_dict,
            self.graph.graph.edge_attr_dict
        )
        
        # ============== Extract Action Node Features ==============
        T = self.traj_horizon
        G = self.graph.num_g_nodes
        P = self.pred_horizon
        
        # Left arm action nodes
        left_time_mask = self.graph.graph.gripper_left_time > T
        x_left = x_dict['gripper_left'][left_time_mask].view(
            self.batch_size, P, G, -1
        )
        
        # Right arm action nodes
        right_time_mask = self.graph.graph.gripper_right_time > T
        x_right = x_dict['gripper_right'][right_time_mask].view(
            self.batch_size, P, G, -1
        )
        
        # ============== Prediction Heads ==============
        # Left arm predictions
        preds_trans_left = self.pred_head_trans_left(x_left)
        preds_rot_left = self.pred_head_rot_left(x_left)
        preds_grip_left = self.pred_head_grip_left(x_left)
        preds_left = torch.cat([preds_trans_left, preds_rot_left, preds_grip_left], dim=-1)
        
        # Right arm predictions
        preds_trans_right = self.pred_head_trans_right(x_right)
        preds_rot_right = self.pred_head_rot_right(x_right)
        preds_grip_right = self.pred_head_grip_right(x_right)
        preds_right = torch.cat([preds_trans_right, preds_rot_right, preds_grip_right], dim=-1)
        
        return preds_left, preds_right
    
    def get_labels(self, 
                   gt_actions: torch.Tensor,
                   noisy_actions: torch.Tensor,
                   gt_grips: torch.Tensor,
                   noisy_grips: torch.Tensor,
                   arm: str) -> torch.Tensor:
        """
        Compute training labels for one arm.
        
        Args:
            gt_actions: [B, P, 4, 4] ground truth actions
            noisy_actions: [B, P, 4, 4] noisy actions
            gt_grips: [B, P, 1] ground truth gripper states
            noisy_grips: [B, P, 1] noisy gripper states
            arm: 'left' or 'right'
            
        Returns:
            labels: [B, P, G, 7] target labels
        """
        gripper_points = self.graph.gripper_keypoints[None, None, :, :].repeat(
            gt_actions.shape[0], gt_actions.shape[1], 1, 1
        )  # [B, P, G, 3]
        
        # Compute relative transform: T_noisy^{-1} @ T_gt
        T_w_n = noisy_actions.view(-1, 4, 4)
        T_n_w = torch.inverse(T_w_n)
        T_w_g = gt_actions.view(-1, 4, 4)
        T_n_g = torch.bmm(T_n_w, T_w_g)
        T_n_g = T_n_g.view(gt_actions.shape[0], gt_actions.shape[1], 4, 4)
        
        # Translation labels
        labels_trans = T_n_g[..., :3, 3][:, :, None, :].repeat(
            1, 1, gripper_points.shape[-2], 1
        )
        
        # Rotation labels (as keypoint offsets)
        T_n_g_rot = T_n_g.clone()
        T_n_g_rot[..., :3, 3] = 0
        labels_rot = self.graph.transform_gripper_nodes(gripper_points, T_n_g_rot) - gripper_points
        
        # Combine
        labels = torch.cat([labels_trans, labels_rot], dim=-1)
        
        # Gripper labels
        labels_grip = gt_grips[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        labels = torch.cat([labels, labels_grip], dim=-1)
        
        return labels
    
    def get_transformed_node_pos(self, actions: torch.Tensor, arm: str) -> torch.Tensor:
        """Get gripper keypoints transformed by actions."""
        gripper_points = self.graph.gripper_keypoints[None, None, :, :].repeat(
            actions.shape[0], actions.shape[1], 1, 1
        )
        return self.graph.transform_gripper_nodes(gripper_points, actions)
