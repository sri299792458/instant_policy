# Bimanual Instant Policy

An extension of [Instant Policy](https://www.robot-learning.uk/instant-policy) for **dual-arm (bimanual) manipulation**.

## Overview

This codebase extends the single-arm Instant Policy to support two cooperating robotic arms with:

- **Dual egocentric frames**: Each arm has its own view of the scene
- **Cross-arm coordination**: Graph attention between arms for synchronized motion
- **Pseudo-demonstration generation**: Synthetic training data with diverse coordination patterns
- **DDIM diffusion**: Denoising for smooth action prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/bimanual_instant_policy.git
cd bimanual_instant_policy

# Create conda environment
conda env create -f environment.yml
conda activate bimanual_ip

# Install the package
pip install -e .
```

## Quick Start

### 1. Generate Pseudo-Demonstrations

```bash
python bimanual_prepare_data.py generate \
    --output_dir ./data/pseudo_demos \
    --num_samples 10000 \
    --split_ratio 0.9
```

### 2. Train the Model

```bash
python bimanual_train.py \
    --run_name bimanual_ip_v1 \
    --batch_size 8 \
    --use_pseudo_demos 1 \
    --record 1
```

### 3. Run Inference

```bash
python bimanual_deployment.py \
    --checkpoint ./checkpoints/bimanual_ip_v1/best.pt \
    --device cuda
```

## Project Structure

```
bimanual_instant_policy/
├── bimanual_graph_rep.py     # Bimanual graph representation
├── bimanual_model.py         # BimanualAGI model
├── bimanual_diffusion.py     # Diffusion training/inference
├── bimanual_dataset.py       # Dataset classes
├── bimanual_train.py         # Training script
├── bimanual_deployment.py    # Inference script
├── bimanual_prepare_data.py  # Data preparation
├── data_structures.py        # Core data types
├── generator.py              # Pseudo-demo generation
├── objects.py                # Procedural objects
├── patterns/                 # Coordination patterns
│   ├── __init__.py
│   └── coordination_patterns.py
├── test_components.py        # Unit tests
├── ip/                       # Original Instant Policy
│   ├── configs/
│   │   ├── base_config.py
│   │   └── bimanual_config.py
│   ├── models/
│   │   ├── diffusion.py
│   │   ├── graph_rep.py
│   │   ├── graph_transformer.py
│   │   ├── model.py
│   │   └── scene_encoder.py
│   └── utils/
│       ├── common_utils.py
│       └── normalizer.py
└── WALKTHROUGH.md            # Detailed documentation
```

## Coordination Patterns

The pseudo-demo generator creates diverse bimanual trajectories:

| Pattern               | Description                         | Weight |
| --------------------- | ----------------------------------- | ------ |
| `symmetric_lift`      | Both arms lift an object together   | 30%    |
| `handover`            | Pass object from one arm to another | 25%    |
| `hold_and_manipulate` | One arm holds, one manipulates      | 25%    |
| `independent`         | Arms work on separate tasks         | 20%    |

## Model Architecture

```
Input (Demos + Observation)
    │
    ▼
Scene Encoder (PointNet++)
    │
    ▼
BimanualGraphRep (Dual Egocentric + Cross-Arm Edges)
    │
    ▼
Local Encoder (Spatial Features)
    │
    ▼
Context Encoder (Demo → Current)
    │
    ▼
Action Encoder (Current → Future + Cross-Arm)
    │
    ▼
DDIM Diffusion (Denoising)
    │
    ▼
Predicted Actions (Left + Right)
```

## Configuration

Key configuration parameters in `ip/configs/bimanual_config.py`:

| Parameter                   | Description               | Default |
| --------------------------- | ------------------------- | ------- |
| `batch_size`                | Training batch size       | 8       |
| `num_demos`                 | Demonstrations per sample | 2       |
| `traj_horizon`              | Waypoints per demo        | 10      |
| `pre_horizon`               | Prediction horizon        | 8       |
| `num_diffusion_iters_train` | Training diffusion steps  | 100     |
| `num_diffusion_iters_test`  | Inference diffusion steps | 8       |
| `use_cross_arm_attention`   | Enable cross-arm edges    | True    |

## Testing

Run component tests:

```bash
python test_components.py
```

## Training Tips

1. **Start with pseudo-demos**: Use synthetic data to get the model working before fine-tuning on real data

2. **Adjust batch size**: Bimanual requires more GPU memory (~2x single-arm)

3. **Monitor both arms**: Track `Val_Trans_Left` and `Val_Trans_Right` separately

4. **Cross-arm importance**: Enable `use_cross_arm_attention` for coordinated tasks

## API Usage

```python
from bimanual_deployment import BimanualPolicy

# Load trained model
policy = BimanualPolicy("./checkpoints/best.pt", device="cuda")

# Load demonstrations
policy.load_demonstrations(demos_left, demos_right, demo_pcds)

# Predict actions
actions_left, grips_left, actions_right, grips_right = policy.predict(
    T_w_left, T_w_right,
    grip_left, grip_right,
    pcd_world
)

# Execute actions
for i in range(len(actions_left)):
    next_pose_left = T_w_left @ actions_left[i]
    next_pose_right = T_w_right @ actions_right[i]
    robot.move_to(next_pose_left, next_pose_right, 
                  grips_left[i], grips_right[i])
```

## Citation

If you use this work, please cite the original Instant Policy:

```bibtex
@inproceedings{instantpolicy,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Author et al.},
  booktitle={ICRA},
  year={2024}
}
```

## License

Apache 2.0 License
