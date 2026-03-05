"""
Generate toy datasets for KAN symbolic regression environment.
Small datasets (150 samples), 2-variable laws, well-conditioned domains.
"""

import numpy as np
import pandas as pd
import json
import os

SEED = 42
N_TRAIN = 150  # toy: small dataset
NOISE_STD = 0.01

# Ground truth laws (hidden from agent, known to judge)
# Format: (law_fn, description, domains, true_expression_template)
LAWS = {
    "A": {
        "fn": lambda x1, x2, k: k * x1 * x2,
        "param_ranges": {"k": (0.5, 3.0)},
        "domains": {"x1": [0.5, 3.0], "x2": [0.5, 3.0]},
        "description": "Product law (physics: e.g. force = k * m * a)",
        "n_vars": 2,
    },
    "B": {
        "fn": lambda x1, x2, k: k * np.sin(x1) + x2**2,
        "param_ranges": {"k": (0.5, 2.0)},
        "domains": {"x1": [0.0, np.pi], "x2": [0.5, 2.0]},
        "description": "Sinusoidal + quadratic (signal processing)",
        "n_vars": 2,
    },
    "C": {
        "fn": lambda x1, x2, k: k * x1 / (x1 + x2),
        "param_ranges": {"k": (1.0, 4.0)},
        "domains": {"x1": [0.5, 3.0], "x2": [0.5, 3.0]},
        "description": "Michaelis-Menten (biology: enzyme kinetics)",
        "n_vars": 2,
    },
    "D": {
        "fn": lambda x1, x2, k: k * np.exp(-x1) * x2,
        "param_ranges": {"k": (0.5, 2.0)},
        "domains": {"x1": [0.1, 2.0], "x2": [0.5, 3.0]},
        "description": "Exponential decay (physics/chemistry)",
        "n_vars": 2,
    },
}


def generate_dataset(dataset_id: str, rng: np.random.Generator, n_samples: int = N_TRAIN):
    law = LAWS[dataset_id]
    domains = law["domains"]
    param_ranges = law["param_ranges"]

    # Sample random constant (unknown to agent)
    k = rng.uniform(*param_ranges["k"])

    # Sample inputs within domain
    x1 = rng.uniform(*domains["x1"], size=n_samples)
    x2 = rng.uniform(*domains["x2"], size=n_samples)

    # Compute outputs
    y_clean = law["fn"](x1, x2, k)
    noise = rng.normal(0, NOISE_STD, size=n_samples)
    y = y_clean + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return df, k


def generate_all(output_dir: str):
    rng = np.random.default_rng(SEED)
    os.makedirs(output_dir, exist_ok=True)

    domains_info = {}
    ground_truth = {}

    for dataset_id, law in LAWS.items():
        df, k = generate_dataset(dataset_id, rng)
        path = os.path.join(output_dir, f"train_{dataset_id}.csv")
        df.to_csv(path, index=False)
        print(f"[{dataset_id}] Saved {len(df)} samples → {path}  (k={k:.4f})")

        domains_info[dataset_id] = law["domains"]
        ground_truth[dataset_id] = {
            "k": round(float(k), 4),
            "description": law["description"],
            "n_vars": law["n_vars"],
        }

    # Write domains.json (visible to agent)
    with open(os.path.join(output_dir, "domains.json"), "w") as f:
        json.dump(domains_info, f, indent=2)
    print(f"\nWrote domains.json")

    # Write ground_truth.json (judge only - NOT in /data/)
    gt_path = os.path.join(output_dir, "..", "judge", "ground_truth.json")
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote ground_truth.json (judge only)")

    return ground_truth


if __name__ == "__main__":
    gt = generate_all("data")
    print("\nGround truth constants (judge only):")
    for did, info in gt.items():
        print(f"  Dataset {did}: k={info['k']}  — {info['description']}")
