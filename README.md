# KAN Symbolic Regression — RL Environment

A reinforcement learning environment where an LLM agent trains Kolmogorov-Arnold Networks (KANs) to discover symbolic scientific laws from noisy data.

Based on the RL environment design from the Preference Model Assessment (KANs + neuro-symbolic reasoning).

---

## Architecture

```
kan-rl-env/
├── env_api/
│   └── kan_env.py          # Instrumented training API (agent MUST use this)
├── datasets/
│   └── generate.py         # Generates 4 toy datasets with hidden laws
├── data/                   # Generated at runtime (gitignored)
│   ├── train_A.csv         # 150 samples, 2 inputs
│   ├── train_B.csv
│   ├── train_C.csv
│   ├── train_D.csv
│   └── domains.json        # Valid input ranges per variable
├── judge/
│   ├── judge.py            # 7-step judge (no false positives by design)
│   └── ground_truth.json   # Generated at runtime, judge-only
├── agent/
│   └── agent.py            # LLM agent using Grok API
├── solution/               # Agent writes here (gitignored except discover.py template)
│   ├── discover.py         # DELIVERABLE: discover_law() + predict()
│   └── models/             # Saved KAN weights
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.sh
```

---

## Datasets (toy model)

| Dataset | Law | Domain | Field |
|---------|-----|--------|-------|
| A | `k * x1 * x2` | x1,x2 ∈ [0.5, 3.0] | Physics (force) |
| B | `k * sin(x1) + x2²` | x1 ∈ [0, π], x2 ∈ [0.5, 2.0] | Signal processing |
| C | `k * x1 / (x1 + x2)` | x1,x2 ∈ [0.5, 3.0] | Biology (Michaelis-Menten) |
| D | `k * exp(-x1) * x2` | x1 ∈ [0.1, 2.0], x2 ∈ [0.5, 3.0] | Chemistry (decay) |

Constants `k` are randomized per run (unknown to agent).

---

## Judge (7 steps)

1. **File & interface check** — discover.py exists, functions importable, smoke test
2. **KAN training verification** — trace from `env_api.train_kan()`, loss improvement, KAN submodules
3. **Expression validity + AST cap** — eval() works, node count ≤ 60/70/80 by dimensionality
4. **Expression ↔ KAN consistency** — 400 points, divergence threshold 0.20 (gating faible)
5. **Multi-OOD generalization** — 3 regimes (IID, OOD-1 ±30%, OOD-2 asymmetric)
6. **Parsimony** — MDL-based complexity penalty, no SymPy normalization
7. **Final score** — `0.60*gen + 0.20*parsimony + 0.10*consistency + 0.10*kan_conformity`

**Pass threshold: 0.65**

---

## Quickstart

### Option 1: Docker (recommended)

```bash
# 1. Copy and fill your API key
cp .env.example .env
# Edit .env: set GROK_API_KEY=xai-...

# 2. Build and run full pipeline (agent + judge)
docker compose --env-file .env up rl-env

# 3. Run judge only (on your own solution)
docker compose up judge

# 4. Generate data only
docker compose up datagen
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets
python3 datasets/generate.py

# Run agent (needs GROK_API_KEY)
export GROK_API_KEY=xai-...
python3 agent/agent.py

# Run judge
python3 judge/judge.py

# Or run everything
bash run.sh full
```

---

## Writing Your Own Solution

The agent must produce `solution/discover.py` with:

```python
def discover_law(dataset_id: str) -> str:
    """Returns numpy-evaluable symbolic expression, e.g. '2.5 * x1 * x2'"""
    ...

def predict(dataset_id: str, X: np.ndarray) -> np.ndarray:
    """Returns KAN model predictions. X shape: (N, 2). Must NOT use expression."""
    ...
```

**You MUST use `env_api.train_kan()` for training** — the judge reads the instrumented trace directly.

```python
from kan import KAN
from env_api.kan_env import train_kan

model = KAN(width=[2, 3, 1], grid=3, k=3, seed=42)
trained_model, trace = train_kan("A", model, {"lr": 0.01, "steps": 150})
```

---

## Reward Hacking Protection

| Attack | Detection |
|--------|-----------|
| Use PySR/gplearn | Import audit (soft −0.30) + expr↔KAN consistency |
| Return expression not from KAN | Step 4: median divergence on 400 points |
| Overfit polynomial | Multi-regime OOD (3 regimes, asymmetric) |
| Fake training logs | env_api traces are written by the API, not the agent |
| Read test data | Ground truth generated inside judge process, never on disk |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROK_API_KEY` | — | Required for agent |
| `GROK_MODEL` | `grok-3-mini` | Grok model to use |
| `MAX_ITERATIONS` | `3` | Agent iteration budget |
| `DATA_DIR` | `data` | Dataset location |
| `SOLUTION_DIR` | `solution` | Agent output location |
| `TRACE_DIR` | `/tmp/kan_traces` | env_api trace location |
