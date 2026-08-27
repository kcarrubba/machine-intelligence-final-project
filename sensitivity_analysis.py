"""
SENSITIVITY ANALYSIS: Experimental Variant Testing

PURPOSE:
This script addresses the rubric requirement for 'evidence from experiments or variants' 
by testing the impact of different hyperparameters on model performance. 

WHAT THIS SCRIPT DOES:
1. Loads the expanded dataset (10k+ images) from 'datasets/FoodTest1_expanded'
2. Creates a 10% stratified subset to enable rapid iterative testing (approx. 20-30 mins)
3. Executes three training variants with different 'Head Learning Rates' (High, Baseline, Low)
4. Saves all outputs to a dedicated 'sensitivity_results/' directory to ensure the 
   primary 'model/best_model.pth' is NEVER overwritten or compromised

USAGE FOR REPORT:
The 'Best validation accuracy' printed at the end of each variant run is used to 
populate 'Table 2: Hyperparameter Sensitivity Analysis' in the final report, 
justifying the selection of the final recommended model configuration
"""

import os
import shutil
import torch
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
import train # Uses your existing training logic

import os
import shutil
import torch
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
import train # Uses your existing training logic

def run_sensitivity_test():
    # 1. Setup
    device = train.get_device()
    output_base = Path("sensitivity_results")
    output_base.mkdir(exist_ok=True)
    
    # 2. Create/Verify the "Mini" Dataset folder (Flat structure)
    mini_data_path = Path("datasets/FoodTest_Mini")
    source_path = Path("datasets/FoodTest1_expanded")
    
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist.")
        return

    if not mini_data_path.exists() or len(list(mini_data_path.glob("*"))) == 0:
        print(f"Creating mini-dataset at {mini_data_path}...")
        mini_data_path.mkdir(parents=True, exist_ok=True)
        
        class_samples = defaultdict(list)
        for f in source_path.iterdir():
            if f.is_file() and not f.name.startswith("."):
                prefix = f.name.split("_")[0]
                if prefix.isdigit() and len(prefix) == 2:
                    class_samples[prefix].append(f)
        
        for prefix, files in class_samples.items():
            for f in files[:10]: # 10 images per class
                shutil.copy(f, mini_data_path / f.name)
        print(f"Mini-dataset created with {len(list(mini_data_path.glob('*')))} images.")

    # 3. Define the experiments (Variants)
    experiments = [
        {"name": "LR_High", "head_lr": 1e-2},
        {"name": "LR_Baseline", "head_lr": 5e-4}, 
        {"name": "LR_Low", "head_lr": 1e-4},
    ]

    for exp in experiments:
        print(f"\n--- Testing Variant: {exp['name']} (Head LR: {exp['head_lr']}) ---")
        
        # We define BOTH 'phasel_epochs' and 'phase1_epochs' to prevent the attribute error
        args = SimpleNamespace(
            data_dir=str(mini_data_path),
            extra_data=None, 
            noise_data=None,
            output_dir=str(output_base / exp['name']),
            batch_size=16,
            phasel_epochs=3, 
            phase1_epochs=3, # Added this to fix the types.SimpleNamespace error
            phase2_epochs=0, 
            val_split=0.2,
            backbone_lr=1e-4,
            head_lr=exp['head_lr']
        )

        try:
            train.run_training(args) 
        except Exception as e:
            print(f"Variant {exp['name']} failed: {e}")

    print("\nAnalysis Complete. Use the Validation Accuracy printed above for your report.")

if __name__ == "__main__":
    run_sensitivity_test()
    