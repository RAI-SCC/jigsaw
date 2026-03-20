import os
import sys
import subprocess
import argparse
from pathlib import Path

TEST_CONFIGS = [
    (256, "fixed"),
    (256, "flex"),
    (512, "fixed"),
    (512, "flex"),
    (768, "fixed"),
    (768, "flex"),
    (1024, "fixed"),
    (1024, "flex"),
]

def run_test(config_path, seed=42, n_visual=6, threshold=0.5):
    cmd = [
        "python", "-u", "scaling_scripts/test_swinmlp.py",
        "--config", config_path,
        "--seed", str(seed),
        "--n_visual", str(n_visual),
        "--threshold", str(threshold),
    ]

    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"Test failed with exit code {result.returncode}")
        return False
    
    print(f"Test completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch test script for all resolutions and kernel types"
    )
    parser.add_argument(
        "--configs_dir",
        default="experiments/swinmlp",
        help="Base directory containing test configs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sample selection (same across all tests)"
    )
    parser.add_argument(
        "--n_visual",
        type=int,
        default=6,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024],
        help="Resolutions to test (default: all)"
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["fixed", "flex"],
        default=["fixed", "flex"],
        help="Kernel types to test (default: both)"
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue with next test if one fails"
    )
    
    args = parser.parse_args()
    
    configs_to_test = [
        (res, kernel) 
        for res, kernel in TEST_CONFIGS 
        if res in args.resolutions and kernel in args.kernels
    ]
    
    if not configs_to_test:
        print("No configurations match the specified filters")
        sys.exit(1)
    
    results = {}
    failed_tests = []
    
    for res, kernel in configs_to_test:
        config_name = f"{res}/config_test_{kernel}.yaml"
        config_path = os.path.join(args.configs_dir, config_name)
        
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            print(f"Skipping {res}_{kernel}")
            results[f"{res}_{kernel}"] = "SKIPPED"
            continue
        
        success = run_test(
            config_path,
            seed=args.seed,
            n_visual=args.n_visual,
            threshold=args.threshold
        )
        
        if success:
            results[f"{res}_{kernel}"] = "SUCCESS"
        else:
            results[f"{res}_{kernel}"] = "FAILED"
            failed_tests.append(f"{res}_{kernel}")
            
            if not args.skip_on_error:
                print(f"Stopping due to failed test: {res}_{kernel}")
                break
    
    if failed_tests:
        print(f"{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        successful = sum(1 for s in results.values() if s == "SUCCESS")
        print(f"All {successful} tests completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()