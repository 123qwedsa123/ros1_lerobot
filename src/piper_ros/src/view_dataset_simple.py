#!/usr/bin/env python3
"""
Simple view of dataset actions.
Prints the first 20 samples to check what actions look like during training.
"""

import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

def view_dataset_actions(dataset_path, num_samples=20):
    """View the first few actions in the dataset"""
    
    root_path = Path(dataset_path)
    # Recursively find file-*.parquet in data directory
    parquet_files = sorted(list(root_path.rglob("data/**/file-*.parquet")))
    
    if not parquet_files:
        # Fallback search if the structure is different
        parquet_files = sorted(list(root_path.rglob("*.parquet")))
        # Filter out meta files
        parquet_files = [f for f in parquet_files if "meta" not in str(f)]
    
    if not parquet_files:
        print(f"❌ No data files found in: {root_path}")
        return
    
    print(f"Reading dataset: {dataset_path}")
    print(f"Data file: {parquet_files[0].relative_to(root_path)}")
    print()
    
    # Read data
    table = pq.read_table(parquet_files[0])
    actions = np.array(table['action'].to_list())
    states = np.array(table['observation.state'].to_list())
    
    print(f"Total Frames: {len(actions)}")
    print(f"Num Joints: {actions.shape[1]}")
    print()
    
    # Print Header
    print("="*110)
    print(f"{'#':<4} | {'Source':<7} | " + " | ".join([f"Joint{i:<3}" for i in range(6)]) + " | Gripper  | Max Diff")
    print("="*110)
    
    # Print samples
    for i in range(min(num_samples, len(actions))):
        state = states[i]
        action = actions[i]
        
        diff = [abs(a - s) for a, s in zip(action, state)]
        max_diff = max(diff)
        
        # State Row
        state_str = f"{i+1:<4} | State   | "
        state_str += " | ".join([f"{s:8.3f}" for s in state])
        print(state_str)
        
        # Action Row
        action_str = f"     | Action  | "
        action_str += " | ".join([f"{a:8.3f}" for a in action])
        print(action_str)
        
        # Diff Row
        diff_str = f"     | Diff    | "
        diff_str += " | ".join([f"{d:8.3f}" for d in diff])
        diff_str += f" | {max_diff:8.3f}"
        
        if max_diff > 0.2:
            print(f"{diff_str}  ⚠️ LARGE")
        else:
            print(f"{diff_str}")
        
        print("-" * 110)
    
    # Summary Statistics
    print("\n" + "="*100)
    print("Summary Statistics (Entire File)")
    print("="*100)
    
    all_diffs = np.abs(actions - states)
    print(f"\n{'Joint':<10} | {'Min Diff':<10} | {'Max Diff':<10} | {'Mean Diff':<10} | {'95th Perc':<10}")
    print("-"*75)
    
    for i in range(actions.shape[1]):
        name = f"Joint {i}" if i < 6 else "Gripper"
        print(f"{name:<10} | {np.min(all_diffs[:,i]):10.4f} | {np.max(all_diffs[:,i]):10.4f} | {np.mean(all_diffs[:,i]):10.4f} | {np.percentile(all_diffs[:,i], 95):10.4f}")
    
    # Consecutive Changes
    action_changes = np.diff(actions, axis=0)
    print("\n" + "="*100)
    print("Consecutive Frame Action Change Statistics")
    print("="*100)
    print(f"\n{'Joint':<10} | {'Max Delta':<10} | {'Mean Delta':<10} | {'95th Perc':<10}")
    print("-"*65)
    
    for i in range(actions.shape[1]):
        name = f"Joint {i}" if i < 6 else "Gripper"
        max_c = np.max(np.abs(action_changes[:, i]))
        print(f"{name:<10} | {max_c:10.4f} | {np.mean(np.abs(action_changes[:,i])):10.4f} | {np.percentile(np.abs(action_changes[:,i]), 95):10.4f}")
        if max_c > 0.3:
            print(f"           ⚠️ {name} has large movement! ({max_c*57.3:.1f} deg)")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./lerobot_dataset"
    view_dataset_actions(path)
