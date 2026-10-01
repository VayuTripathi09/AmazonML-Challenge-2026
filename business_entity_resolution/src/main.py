"""
Main Entrypoint for Business Entity Resolution System.
Supports CLI modes:
  python src/main.py --mode profile
  python src/main.py --mode train
  python src/main.py --mode validate
  python src/main.py --mode predict
  python src/main.py --mode all
"""

import sys
import os
import time

# Ensure src directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from config import Config, parse_args
from profiling import run_profiling
from validation import run_validation_pipeline
from inference import run_test_inference

def main():
    args = parse_args()
    
    config = Config(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        decision_threshold=args.threshold,
        max_candidates_per_s1=args.max_candidates_per_s1,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 80)
    print("BUSINESS ENTITY RESOLUTION SYSTEM")
    print(f"Execution Mode:     {args.mode}")
    print(f"Train Dataset Dir:  {config.train_dir}")
    print(f"Test Dataset Dir:   {config.test_dir}")
    print(f"Output Directory:   {config.output_dir}")
    print(f"Decision Threshold: {config.decision_threshold}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    model = None
    
    if args.mode in ["profile", "all"]:
        run_profiling(config)
        
    if args.mode in ["train", "validate", "all"]:
        model, best_th, val_results = run_validation_pipeline(config)
        config.decision_threshold = best_th
        
    if args.mode in ["predict", "all"]:
        run_test_inference(config, model=model)
        
    total_elapsed = time.time() - start_time
    print(f"\nExecution mode '{args.mode}' completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes).")

if __name__ == '__main__':
    main()
