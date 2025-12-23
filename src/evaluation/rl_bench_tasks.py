"""
Bimanual RLBench task mappings.

Maps friendly task names to bimanual task classes from PerAct2's RLBench extension.
"""
from rlbench.utils import name_to_task_class

# Bimanual tasks from PerAct2 RLBench extension
# Use name_to_task_class with bimanual=True to load from rlbench.bimanual_tasks

BIMANUAL_TASK_NAMES = {
    # Coordinated tasks (tight coupling required)
    'lift_tray': 'bimanual_lift_tray',
    'lift_ball': 'bimanual_lift_ball',
    'close_jar': 'coordinated_close_jar',
    'lift_stick': 'coordinated_lift_stick',
    'take_shoes': 'coordinated_take_shoes_out_of_box',
    
    # Handover tasks (one arm passes to other)
    'handover': 'bimanual_handover_item',
    'handover_easy': 'bimanual_handover_item_easy',
    'handover_medium': 'handover_item_medium',
    
    # Dual manipulation (both arms act independently but simultaneously)
    'dual_buttons': 'bimanual_dual_push_buttons',
    'push_box': 'bimanual_push_box',
    'set_table': 'bimanual_set_the_table',
    
    # Sequential tasks (arms take turns)
    'pick_laptop': 'bimanual_pick_laptop',
    'pick_plate': 'bimanual_pick_plate',
    'put_bottle_fridge': 'bimanual_put_bottle_in_fridge',
    'put_item_drawer': 'bimanual_put_item_in_drawer',
    'take_tray_oven': 'bimanual_take_tray_out_of_oven',
    'sweep_dustpan': 'bimanual_sweep_to_dustpan',
    'straighten_rope': 'bimanual_straighten_rope',
    
    # Coordinated drawer tasks
    'put_drawer_right': 'coordinated_put_item_in_drawer_right',
}


def get_bimanual_task_class(task_name: str):
    """
    Get the task class for a bimanual task.
    
    Args:
        task_name: Friendly name from BIMANUAL_TASK_NAMES or full task file name
        
    Returns:
        Task class that can be instantiated
        
    Raises:
        ValueError: If task_name is not found
    """
    # Check if it's a bimanual task
    if task_name in BIMANUAL_TASK_NAMES:
        task_file = BIMANUAL_TASK_NAMES[task_name]
    else:
        # Assume it's the actual task file name - try bimanual first
        task_file = task_name
    
    try:
        return name_to_task_class(task_file, bimanual=True)
    except Exception as e:
        raise ValueError(
            f"Could not load bimanual task '{task_name}' (file: {task_file}). "
            f"Available bimanual tasks: {list(BIMANUAL_TASK_NAMES.keys())}."
        ) from e
