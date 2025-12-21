"""
Bimanual Graph Diffusion for dual-arm action prediction.

Extends the single-arm GraphDiffusion to handle bimanual manipulation
with separate diffusion processes for each arm.
"""
import lightning as L
import torch
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
import warnings

from ip.utils.common_utils import (
    transforms_to_actions,
    actions_to_transforms,
    rotation_matrix_to_angle_axis,
    get_rigid_transforms,
)
from ip.utils.normalizer import Normalizer
from ip.utils.repairs import repair_checkpoint
from bimanual_model import BimanualAGI

warnings.filterwarnings("ignore", category=UserWarning)


class BimanualGraphDiffusion(L.LightningModule):
    """
    Bimanual Graph Diffusion model for dual-arm manipulation.
    
    Performs DDIM diffusion on actions for both arms simultaneously,
    with cross-arm coordination through the graph transformer.
    """
    
    def __init__(self, config):
        super().__init__()
        self.model = BimanualAGI(config)
        
        # Store references for parameter counting
        self.graph_rep = self.model.graph
        self.scene_encoder = self.model.scene_encoder
        # BimanualAGI has three separate encoders, not one unified encoder
        self.local_encoder = self.model.local_encoder
        self.cond_encoder = self.model.cond_encoder
        self.action_encoder = self.model.action_encoder
        
        self.config = config
        self.record = config.get('record', False)
        self.save_dir = config.get('save_dir', None)
        self.save_every = config.get('save_every', 100000)
        self.randomise_num_demos = config.get('randomise_num_demos', False)
        self.use_lr_scheduler = config.get('use_lr_scheduler', False)
        
        # Tracking
        self.best_trans_loss = 1e6
        self.val_losses = []
        
        # DDIM Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config['num_diffusion_iters_train'],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            prediction_type='sample',
        )
        
        self.loss_fn = torch.nn.L1Loss()
        
        # Normalizers for each arm (same normalization)
        self.normalizer_left = Normalizer(
            pred_horizon=config['pre_horizon'],
            min_action=config['min_actions'].to(config['device']),
            max_action=config['max_actions'].to(config['device']),
            device=config['device']
        )
        self.normalizer_right = Normalizer(
            pred_horizon=config['pre_horizon'],
            min_action=config['min_actions'].to(config['device']),
            max_action=config['max_actions'].to(config['device']),
            device=config['device']
        )
    
    def add_noise(self, actions: torch.Tensor, grip_actions: torch.Tensor,
                  timesteps: torch.Tensor, normalizer: Normalizer):
        """
        Add noise to actions for diffusion training.
        
        Args:
            actions: [B, P, 4, 4] SE(3) actions
            grip_actions: [B, P] gripper states
            timesteps: [B] diffusion timesteps
            normalizer: Normalizer for this arm
            
        Returns:
            noisy_actions: [B, P, 4, 4]
            noisy_grips: [B, P]
        """
        b, p = actions.shape[:2]
        
        # Convert to 6D (translation + angle-axis)
        actions_6d = transforms_to_actions(actions.view(-1, 4, 4)).view(b, p, 6)
        
        # Normalize
        actions_6d = normalizer.normalize_actions(actions_6d)
        
        # Add noise
        noise = torch.randn_like(actions_6d)
        noisy_actions = self.noise_scheduler.add_noise(actions_6d, noise, timesteps)
        noisy_actions = torch.clamp(noisy_actions, -1, 1)
        
        # Denormalize
        noisy_actions = normalizer.denormalize_actions(noisy_actions)
        
        # Convert back to SE(3)
        noisy_actions = actions_to_transforms(noisy_actions.view(-1, 6)).view(b, p, 4, 4)
        
        # Add noise to gripper
        noise_g = torch.randn_like(grip_actions.unsqueeze(-1))
        noisy_grips = self.noise_scheduler.add_noise(
            grip_actions.unsqueeze(-1), noise_g, timesteps
        ).squeeze(-1)
        noisy_grips = torch.clamp(noisy_grips, -1, 1)
        
        return noisy_actions, noisy_grips
    
    def se3_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        """Compute SE(3) loss between predicted and ground truth poses."""
        # Translation error
        trans_err = torch.norm(pred[..., :3, 3] - gt[..., :3, 3], dim=-1).mean()
        
        # Rotation error
        rot_error = torch.eye(4, device=pred.device).repeat(
            pred.shape[0], pred.shape[1], 1, 1
        )
        rot_error[..., :3, :3] = pred[..., :3, :3].transpose(-1, -2) @ gt[..., :3, :3]
        rot_error = rot_error.view(-1, 4, 4)
        angle_axis = rotation_matrix_to_angle_axis(rot_error[:, :3, :])
        rot_err = angle_axis.norm(dim=-1).mean() * 180 / np.pi
        
        return trans_err, rot_err
    
    def training_step(self, data, batch_idx):
        batch_size = data.actions_left.shape[0]
        
        if self.randomise_num_demos:
            num_demos = np.random.randint(1, self.config['num_demos'] + 1)
            self.model.reinit_graphs(batch_size, num_demos=num_demos)
        
        # Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to both arms
        noisy_actions_left, noisy_grips_left = self.add_noise(
            data.actions_left, data.actions_grip_left, timesteps, self.normalizer_left
        )
        noisy_actions_right, noisy_grips_right = self.add_noise(
            data.actions_right, data.actions_grip_right, timesteps, self.normalizer_right
        )
        
        # Compute labels for both arms
        labels_left = self.model.get_labels(
            data.actions_left, noisy_actions_left,
            data.actions_grip_left.unsqueeze(-1),
            noisy_grips_left.unsqueeze(-1),
            'left'
        )
        labels_right = self.model.get_labels(
            data.actions_right, noisy_actions_right,
            data.actions_grip_right.unsqueeze(-1),
            noisy_grips_right.unsqueeze(-1),
            'right'
        )
        
        # Normalize labels
        labels_left[..., :6] = self.normalizer_left.normalize_labels(labels_left[..., :6])
        labels_right[..., :6] = self.normalizer_right.normalize_labels(labels_right[..., :6])
        
        # CRITICAL: Save ground truth before overwriting!
        # Coordination loss needs GT, not noisy actions
        gt_actions_left = data.actions_left.clone()
        gt_actions_right = data.actions_right.clone()
        
        # Store noisy actions in data
        data.actions_left = noisy_actions_left
        data.actions_right = noisy_actions_right
        data.actions_grip_left = noisy_grips_left
        data.actions_grip_right = noisy_grips_right
        data.diff_time = timesteps.view(-1, 1)
        
        # Forward pass
        preds_left, preds_right = self.model(data)
        
        # Compute loss
        loss_left = self.loss_fn(preds_left, labels_left)
        loss_right = self.loss_fn(preds_right, labels_right)
        loss = (loss_left + loss_right) / 2
        
        # Optional coordination consistency loss
        # CRITICAL: Must compute on MODEL PREDICTIONS, not noisy inputs!
        if self.config.get('use_coordination_loss', False):
            # Decode predictions from delta format to SE(3) actions
            pred_actions_left = self._decode_predictions_to_actions(
                preds_left, noisy_actions_left, 'left'
            )
            pred_actions_right = self._decode_predictions_to_actions(
                preds_right, noisy_actions_right, 'right'
            )
            
            # Now compute coordination loss on predictions vs ground truth
            coord_loss = self._compute_coordination_loss(
                pred_actions_left, pred_actions_right,
                gt_actions_left, gt_actions_right
            )
            coord_weight = self.config.get('coordination_loss_weight', 0.1)
            loss = loss + coord_weight * coord_loss
            self.log("Train_Coord_Loss", coord_loss, on_step=False, on_epoch=True)
        
        self.log("Train_Loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Train_Loss_Left", loss_left, on_step=False, on_epoch=True)
        self.log("Train_Loss_Right", loss_right, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, data, batch_idx, vis=False, ret_actions=False):
        batch_size = data.actions_left.shape[0]
        self.model.reinit_graphs(batch_size, num_demos=self.config['num_demos_test'])
        
        gt_actions_left = data.actions_left.clone()
        gt_actions_right = data.actions_right.clone()
        gt_grips_left = data.actions_grip_left.clone()
        gt_grips_right = data.actions_grip_right.clone()
        
        with torch.autocast(dtype=torch.float32, device_type='cuda'):
            actions_left, grips_left, actions_right, grips_right = self.test_step(
                data, batch_idx, vis=vis
            )
        
        # Compute errors
        trans_err_left, rot_err_left = self.se3_loss(actions_left, gt_actions_left)
        trans_err_right, rot_err_right = self.se3_loss(actions_right, gt_actions_right)
        
        grip_loss_left = (grips_left.squeeze() - gt_grips_left).abs().mean()
        grip_loss_right = (grips_right.squeeze() - gt_grips_right).abs().mean()
        
        # Log
        self.log("Val_Trans_Left", trans_err_left, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val_Trans_Right", trans_err_right, on_step=False, on_epoch=True)
        self.log("Val_Rot_Left", rot_err_left, on_step=False, on_epoch=True)
        self.log("Val_Rot_Right", rot_err_right, on_step=False, on_epoch=True)
        self.log("Val_Grip_Left", grip_loss_left, on_step=False, on_epoch=True)
        self.log("Val_Grip_Right", grip_loss_right, on_step=False, on_epoch=True)
        
        mean_trans_err = (trans_err_left + trans_err_right) / 2
        self.val_losses.append(mean_trans_err)
        
        self.model.reinit_graphs(self.config['batch_size'], num_demos=self.config['num_demos'])
        
        if ret_actions:
            return actions_left, grips_left, actions_right, grips_right
        return 0
    
    def test_step(self, data, batch_idx, vis=False):
        """Run diffusion inference to predict actions."""
        batch_size = data.actions_left.shape[0]
        P = self.config['pre_horizon']
        
        # Initialize with random noise
        noisy_actions_left = torch.randn((batch_size, P, 6), device=self.device)
        noisy_actions_left = torch.clamp(noisy_actions_left, -1, 1)
        noisy_actions_left = self.normalizer_left.denormalize_actions(noisy_actions_left)
        noisy_actions_left = actions_to_transforms(
            noisy_actions_left.view(-1, 6)
        ).view(batch_size, P, 4, 4)
        
        noisy_actions_right = torch.randn((batch_size, P, 6), device=self.device)
        noisy_actions_right = torch.clamp(noisy_actions_right, -1, 1)
        noisy_actions_right = self.normalizer_right.denormalize_actions(noisy_actions_right)
        noisy_actions_right = actions_to_transforms(
            noisy_actions_right.view(-1, 6)
        ).view(batch_size, P, 4, 4)
        
        noisy_grips_left = torch.randn((batch_size, P, 1), device=self.device)
        noisy_grips_left = torch.clamp(noisy_grips_left, -1, 1)
        
        noisy_grips_right = torch.randn((batch_size, P, 1), device=self.device)
        noisy_grips_right = torch.clamp(noisy_grips_right, -1, 1)
        
        # Set timesteps
        self.noise_scheduler.set_timesteps(self.config['num_diffusion_iters_test'])
        
        # Diffusion loop
        for k in range(self.config['num_diffusion_iters_test'] - 1, -1, -1):
            # Update data with current noisy actions
            data.actions_left = noisy_actions_left
            data.actions_right = noisy_actions_right
            data.actions_grip_left = noisy_grips_left.squeeze(-1)
            data.actions_grip_right = noisy_grips_right.squeeze(-1)
            
            # Timestep
            if k != self.config['num_diffusion_iters_test'] - 1:
                t = k
            else:
                t = self.config['num_diffusion_iters_train']
            data.diff_time = torch.tensor([[t]] * batch_size, device=self.device)
            
            # Forward pass
            preds_left, preds_right = self.model(data)
            
            # Denormalize predictions
            preds_left[..., :6] = self.normalizer_left.denormalize_labels(preds_left[..., :6])
            preds_right[..., :6] = self.normalizer_right.denormalize_labels(preds_right[..., :6])
            
            # Update actions using predicted deltas
            noisy_actions_left, noisy_grips_left = self._diffusion_step(
                noisy_actions_left, noisy_grips_left, preds_left, k, 
                self.normalizer_left, 'left'
            )
            noisy_actions_right, noisy_grips_right = self._diffusion_step(
                noisy_actions_right, noisy_grips_right, preds_right, k,
                self.normalizer_right, 'right'
            )
        
        return noisy_actions_left, torch.sign(noisy_grips_left), \
               noisy_actions_right, torch.sign(noisy_grips_right)
    
    def _diffusion_step(self, noisy_actions, noisy_grips, preds, k, normalizer, arm):
        """Perform one diffusion step."""
        batch_size = noisy_actions.shape[0]
        
        # Get current gripper positions
        current_gripper = self.model.get_transformed_node_pos(
            noisy_actions, arm, transform=False
        )
        
        # Compute predicted gripper positions
        pred_output = preds[..., 3:6] + current_gripper + \
                      torch.mean(preds[..., :3], dim=-2, keepdim=True)
        
        # Diffusion step for positions
        pred_gripper = self.noise_scheduler.step(
            model_output=pred_output,
            sample=current_gripper,
            timestep=k,
        ).prev_sample
        
        # Get rigid transformation
        T_e_e = get_rigid_transforms(
            current_gripper.view(-1, pred_gripper.shape[-2], 3),
            pred_gripper.view(-1, pred_gripper.shape[-2], 3)
        ).view(batch_size, -1, 4, 4)
        
        noisy_actions = torch.matmul(noisy_actions, T_e_e)
        
        # Diffusion step for gripper
        noisy_grips = self.noise_scheduler.step(
            model_output=preds[..., -1:].mean(dim=-2),
            sample=noisy_grips,
            timestep=k,
        ).prev_sample
        noisy_grips = torch.clamp(noisy_grips, -1, 1)
        
        # Normalize, clamp, denormalize actions
        noisy_actions_6d = transforms_to_actions(
            noisy_actions.view(-1, 4, 4)
        ).view(batch_size, -1, 6)
        noisy_actions_6d = normalizer.normalize_actions(noisy_actions_6d)
        noisy_actions_6d = torch.clamp(noisy_actions_6d, -1, 1)
        noisy_actions_6d = normalizer.denormalize_actions(noisy_actions_6d)
        noisy_actions = actions_to_transforms(
            noisy_actions_6d.view(-1, 6)
        ).view(batch_size, -1, 4, 4)
        
        return noisy_actions, noisy_grips
    
    def _compute_coordination_loss(self, noisy_left: torch.Tensor, noisy_right: torch.Tensor,
                                    gt_left: torch.Tensor, gt_right: torch.Tensor) -> torch.Tensor:
        """
        Compute coordination consistency loss to maintain relative poses.
        
        Penalizes large changes in the relative transformation between arms,
        encouraging coordinated movement during symmetric/cooperative tasks.
        
        Args:
            noisy_left: [B, P, 4, 4] noisy left actions
            noisy_right: [B, P, 4, 4] noisy right actions
            gt_left: [B, P, 4, 4] ground truth left actions
            gt_right: [B, P, 4, 4] ground truth right actions
        
        Returns:
            Scalar coordination loss
        """
        B, P = noisy_left.shape[:2]
        
        # Compute relative transforms: T_left_to_right = T_left^-1 @ T_right
        # For noisy actions
        T_left_inv_noisy = torch.inverse(noisy_left.reshape(-1, 4, 4))
        T_right_noisy = noisy_right.reshape(-1, 4, 4)
        T_rel_noisy = torch.bmm(T_left_inv_noisy, T_right_noisy).reshape(B, P, 4, 4)
        
        # For ground truth actions
        T_left_inv_gt = torch.inverse(gt_left.reshape(-1, 4, 4))
        T_right_gt = gt_right.reshape(-1, 4, 4)
        T_rel_gt = torch.bmm(T_left_inv_gt, T_right_gt).reshape(B, P, 4, 4)
        
        # Loss on relative translation
        trans_loss = torch.norm(T_rel_noisy[..., :3, 3] - T_rel_gt[..., :3, 3], dim=-1).mean()
        
        # Loss on relative rotation (Frobenius norm of rotation matrix difference)
        rot_diff = T_rel_noisy[..., :3, :3] - T_rel_gt[..., :3, :3]
        rot_loss = torch.norm(rot_diff.reshape(B, P, 9), dim=-1).mean()
        
        return trans_loss + rot_loss
    
    def _decode_predictions_to_actions(self, preds: torch.Tensor, 
                                       noisy_actions: torch.Tensor,
                                       arm: str) -> torch.Tensor:
        """
        Decode model predictions from delta format back to SE(3) actions.
        
        The model predicts deltas/flows relative to gripper keypoints.
        This reverses get_labels() to reconstruct predicted actions.
        
        Args:
            preds: [B, P, G, 7] model predictions (trans delta, rot delta, grip)
            noisy_actions: [B, P, 4, 4] noisy input actions
            arm: 'left' or 'right'
            
        Returns:
            pred_actions: [B, P, 4, 4] reconstructed SE(3) actions
        """
        B, P, G = preds.shape[:3]
        
        # Get gripper keypoints
        gripper_kp = self.model.graph.gripper_keypoints[None, None, :, :].repeat(B, P, 1, 1)  # [B, P, G, 3]
        
        # Denormalize predictions (they were normalized during training)
        preds_denorm = preds.clone()
        normalizer = self.normalizer_left if arm == 'left' else self.normalizer_right
        preds_denorm[..., :6] = normalizer.denormalize_labels(preds_denorm[..., :6])
        
        # Extract components
        pred_trans = preds_denorm[..., :3]      # [B, P, G, 3]
        pred_rot = preds_denorm[..., 3:6]       # [B, P, G, 3]
        
        # Translation: average translation across keypoints
        translation = pred_trans.mean(dim=-2)    # [B, P, 3]
        
        # Rotation: Use rigid transform from keypoint offsets
        # pred_rot gives us where keypoints should move
        # Match diffusion update: add translation before rigid fit
        from ip.utils.common_utils import get_rigid_transforms
        
        current_kp = gripper_kp  # [B, P, G, 3]
        target_kp = current_kp + pred_rot + translation[:, :, None, :]  # [B, P, G, 3]
        
        # Get rigid transform that maps current_kp -> target_kp
        T_delta = get_rigid_transforms(
            current_kp.reshape(-1, G, 3),
            target_kp.reshape(-1, G, 3)
        ).reshape(B, P, 4, 4)
        
        # Combine: The predicted action is noisy_action composed with delta
        # pred_action = noisy_action @ T_delta
        pred_actions = torch.bmm(
            noisy_actions.reshape(-1, 4, 4),
            T_delta.reshape(-1, 4, 4)
        ).reshape(B, P, 4, 4)
        
        return pred_actions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        if self.use_lr_scheduler:
            lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=self.config.get('num_warmup_steps', 1000),
                num_training_steps=self.config.get('num_iters', 50000000),
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        
        return optimizer
    
    def on_validation_epoch_end(self, *args, **kwargs):
        if not self.val_losses:
            return
            
        mean_trans_err = torch.tensor(self.val_losses).mean()
        self.val_losses = []
        
        if self.best_trans_loss > mean_trans_err and self.record:
            self.save_model(f'{self.save_dir}/best.pt')
            self.best_trans_loss = mean_trans_err
    
    def save_model(self, path, save_compiled=False):
        self.trainer.save_checkpoint(path)
        if self.config.get('compile_models', False):
            repair_checkpoint(path, save_path=path)
            if save_compiled:
                path_compiled = path.replace('.pt', '_compiled.pt')
                self.trainer.save_checkpoint(path_compiled)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step % self.save_every == 0 and self.record:
            self.save_model(f'{self.save_dir}/{self.global_step}.pt', save_compiled=False)
    
    def on_train_epoch_end(self, *args, **kwargs):
        if self.record:
            self.save_model(f'{self.save_dir}/last.pt', save_compiled=True)
