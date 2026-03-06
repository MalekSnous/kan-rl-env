# kan-rl-env

**An LLM agent trained via a Reinforcement Learning loop to train Kolmogorov-Arnold Networks (KANs) for symbolic scientific law discovery.**

> The agent iteratively improves its KAN training strategy across 20 RL rounds — receiving structured feedback on generalization, expression complexity, and OOD extrapolation — until it discovers interpretable mathematical laws from noisy scientific datasets.

---

## What this is

Standard ML pipelines are designed by humans. This project inverts that: an LLM agent **designs the pipeline itself**, adapting its choices round after round based on structured reward signals.

The task is symbolic regression on four scientific datasets drawn from physics, biology, chemistry, and neuroscience. Each dataset is generated from an unknown law with Gaussian noise. The agent must:

1. Choose a KAN architecture (`width`, `grid`, `k`)
2. Configure training hyperparameters (`lr`, `steps`, `lambda_reg`)
3. Select a symbolic function library for activation identification
4. Extract a human-readable mathematical expression consistent with the trained KAN

The deliverable is simultaneously a **working trained KAN** and a **symbolic expression** that the judge verifies against the KAN's own learned activations — not against ground truth. This dual constraint is what makes reward hacking structurally difficult.

---

## The RL Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                         RL ROUND N                              │
│                                                                 │
│  ┌──────────┐    code + configs    ┌──────────────────────┐    │
│  │          │ ──────────────────▶  │                      │    │
│  │   LLM    │                      │  KAN Training        │    │
│  │  Agent   │                      │  (env_api)           │    │
│  │          │ ◀──────────────────  │                      │    │
│  └──────────┘   structured         └──────────┬───────────┘    │
│       ▲         feedback                       │                │
│       │                                        ▼                │
│       │                            ┌──────────────────────┐    │
│       └────────────────────────────│       Judge          │    │
│         score + per-dataset        │  (7-step evaluation) │    │
│         diagnosis                  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

Each round, the agent receives:
- A **per-dataset score** (generalization × parsimony × consistency × KAN conformity)
- **Diagnostic signals**: stale model detection, OOD-2 collapse warnings, divergence alerts, parsimony gap
- A **best_config** history: the exact hyperparameters that produced the best score per dataset
- A **priority order**: which datasets to focus on first

The agent analyzes this feedback, writes new Python code, and submits an updated solution. No gradient flows through the LLM — the improvement signal is purely textual, making this a genuine RL-over-language setup.

---

## A Concrete Feedback Example — Round 10 → 11

**Round 10** scores 0.61 — all four datasets flagged as stale models:

```
🟡 NEEDS IMPROVEMENT D: 0.4698  — STALE MODEL detected. Always retrain all 4 datasets each round.
🟢 ACCEPTABLE B:        0.5765  — STALE MODEL detected.
🟢 ACCEPTABLE A:        0.6572  — STALE MODEL detected.
🟢 ACCEPTABLE C:        0.7366  — STALE MODEL detected.
```

The feedback system detects that the agent reused cached models instead of retraining — `initial_loss` identical to a previous round gives it away. The judge also notes that D dropped from its best of **0.8436** (round 9) to 0.4698.

**Round 11** — the agent responds:

```python
configs = {
    'A': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.001},
    'B': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.002},  # λ ↑ to reduce overfit
    'C': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.001},
    'D': {'width': [2,2,1], 'steps': 400, 'lambda_reg': 0.001},  # narrower net
}
```

Dataset D with `[2,2,1]` produces an expression with only **43 AST nodes** — well below the cap — scoring `parsimony=0.458`. Final score: **0.6527 ✅ PASS**.

```
Dataset D expression: -25.15*np.sin(0.1733*x1 - 0.1072*x2 + 1.379) + 25.39
AST nodes: 43  →  parsimony = exp(-0.03 × 26) = 0.458
```

This is the feedback loop at work: a stale model warning + best_config hint → architecture change → parsimony bonus unlocked → threshold crossed.

---

## Results

### GPT-OSS 120B — 11 rounds

| Round | Final Score | A | B | C | D | Status |
|-------|------------|---|---|---|---|--------|
| 1 | 0.5751 | 0.657 | 0.602 | 0.572 | 0.469 | ❌ |
| 4 | 0.5591 | 0.657 | 0.577 | 0.565 | 0.437 | ❌ |
| 5 | 0.5894 | 0.657 | 0.577 | 0.551 | 0.573 | ❌ |
| 6 | 0.5985 | 0.657 | 0.603 | 0.664 | 0.470 | ❌ |
| 7 | 0.5989 | 0.657 | 0.537 | **0.737** | 0.465 | ❌ |
| 9 | 0.6234 | 0.657 | 0.221 | 0.772 | **0.844** | ❌ |
| 10 | 0.6100 | 0.657 | 0.577 | 0.737 | 0.470 | ❌ |
| **11** | **0.6527** | **0.657** | **0.571** | **0.737** | **0.646** | **✅ PASS** |

**Best per-dataset across all rounds:**

| Dataset | Best Score | Best Gen | Best OOD-2 | Best Parsimony |
|---------|-----------|----------|------------|----------------|
| A | 0.657 | 0.762 | 0.669 | 0.000 |
| B | 0.603 | 0.672 | 0.516 | 0.181 |
| C | **0.772** | **0.894** | **0.908** | 0.533 |
| D | **0.844** | **0.949** | **0.955** | 0.458 |

C and D reach strong generalization scores (gen > 0.89) with solid OOD-2 extrapolation. A is systematically limited by expression complexity (81 nodes, 1 above cap). B remains the weakest dataset — persistent overfit on OOD-2.

---

## Architecture

```
kan-rl-env/
├── env_api/
│   └── kan_env.py          # Instrumented training API (train_kan, safe_refine, safe_sqrt)
├── judge/
│   └── judge.py            # 7-step scoring judge
├── agent/
│   └── agent.py            # LLM agent (OpenRouter-compatible)
├── feedback_formatter.py   # Converts judge output → structured RL feedback
├── log_results.py          # Results logger + matplotlib plots
├── data/                   # Scientific datasets (A, B, C, D)
├── solution/               # Agent-written solution (discover.py + models/)
└── docker-compose.yml
```

---

## Scoring

```
score = 0.60 × generalization
      + 0.20 × parsimony
      + 0.10 × consistency
      + 0.10 × kan_conformity

final_score = mean(A, B, C, D)   ←  pass threshold: 0.65
```

**Generalization** is evaluated across 3 OOD regimes:
- IID (0.40): new points in the training domain
- OOD-1 (0.35): all variables extended ±30%
- OOD-2 (0.25): asymmetric — x1 at 2× range (hardest, targets wrong functional forms)

**Parsimony** uses an exponential decay with a hard cap:
```
parsimony = exp(-0.03 × complexity)   if ast_nodes ≤ 80
parsimony = 0                          if ast_nodes > 80

complexity = n_ops + 2 × n_constants
```

**Consistency** verifies that `discover_law()` and `predict()` agree on 400 held-out points — catching agents that return a hardcoded expression without a real KAN.

**KAN Conformity** verifies training traces, model hashes, and spline submodule presence via the instrumented `env_api` — cannot be spoofed from log files.

---

## Reward Hacking Prevention

The triple constraint makes single-axis gaming ineffective:

| Attack | Detection |
|--------|-----------|
| Use PySR/gplearn | Import audit + expr↔KAN consistency check |
| Hardcode expression | `expr(X)` vs `predict(X)` on 400 held-out points |
| Overfit polynomial passing OOD-1 | Asymmetric OOD-2 + AST hard cap |
| Read test data from disk | Ground truth generated inside judge at runtime |
| Fake training trace | `env_api` instruments automatically, `model_hash` verified |

---

## Quickstart

```bash
# Configure model and API key
cp .env.example .env
# OPENROUTER_API_KEY=...
# AGENT_MODEL=openai/gpt-oss-120b   # or meta-llama/llama-3.3-70b-versatile

# Run 20 RL rounds
make restart MODEL="gpt-oss-120b"

# Log and plot results
make logs MODEL="gpt-oss-120b"
python log_results.py --plot --summary
```

**Requirements:** Docker, Docker Compose. Python 3.11+ on host for plotting (`pip install matplotlib numpy`).

---

## Logging & Comparison

```bash
# After each run, parse logs into structured JSON
python log_results.py --model "gpt-oss-120b"   --logfile logs/run_gpt.txt
python log_results.py --model "llama-3.3-70b"  --logfile logs/run_llama70b.txt
python log_results.py --model "llama-3.1-8b"   --logfile logs/run_llama8b.txt

# Generate comparison plots
python log_results.py --plot --summary
```

Produces 5 plots: score evolution, per-dataset evolution, best score comparison, OOD-2 extrapolation quality, and parsimony evolution across models.

---

## Environment Design Notes

**Why KANs?** KANs (Liu et al., 2024) replace fixed activation functions with learnable splines on edges — making learned representations directly inspectable and symbolically identifiable. This makes them uniquely suited for scientific law discovery, where the goal is not just prediction but understanding.

**Why symbolic regression?** It creates a verifiable dual deliverable: the agent must produce both a working model and a human-readable expression that the judge can verify for mutual consistency — without access to ground truth. This closes the main reward hacking avenue that single-objective judges leave open.

**Why RL over text?** Each round is a full experiment: the agent writes code, trains a KAN, receives a structured score, and adapts. This is a microcosm of how ML research actually works — iterative, signal-driven, with a fixed budget of experiments.

---

## References

- Liu, Z. et al. (2024). [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- Cranmer, M. (2023). [Interpretable Machine Learning for Science](https://arxiv.org/abs/2301.04589)