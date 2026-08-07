import os
import sys
import argparse
import pandas as pd

# Add the telemetry directory to python path so simulate_alloc can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulate_alloc import generate_trace

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic allocation dataset")
    parser.add_argument("--out-dir", type=str, default="traces/dataset", help="Output directory")
    parser.add_argument("--n-seeds", type=int, default=25, help="Number of seeds per category")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []

    categories = ["clean", "leaky", "spiky"]

    print(f"Generating dataset with {args.n_seeds} seeds per pattern across [{', '.join(categories)}]...")

    for category in categories:
        for seed in range(args.n_seeds):
            filename = f"{category}_seed_{seed}.csv"
            filepath = os.path.join(args.out_dir, filename)
            
            # Generate trace using simulate_alloc generator
            df = generate_trace(pattern=category, seed=seed)
            df.to_csv(filepath, index=False)

            manifest.append({
                "filename": filename,
                "pattern": category,
                "seed": seed,
                "num_events": len(df)
            })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Successfully generated {len(manifest)} traces and saved manifest to {manifest_path}")

if __name__ == "__main__":
    main()
