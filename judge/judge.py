"""
judge/judge.py
Complete judge implementation for KAN symbolic regression RL environment.
Implements all 7 steps from the proposal with corrections applied.
"""

import ast
import importlib.util
import json
import math
import os
import sys
import traceback
from typing import Optional

import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
SOLUTION_DIR = os.environ.get("SOLUTION_DIR", "solution")
DATA_DIR = os.environ.get("DATA_DIR", "data")
TRACE_DIR = os.environ.get("TRACE_DIR", "/tmp/kan_traces")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

DATASET_IDS = ["A", "B", "C", "D"]
PASS_THRESHOLD = 0.65

# AST caps by n_vars
AST_CAPS = {2: 60, 3: 70, 4: 80}

# Weights
W_GENERALIZATION = 0.60
W_PARSIMONY = 0.20
W_CONSISTENCY = 0.10
W_KAN_CONFORMITY = 0.10

FORBIDDEN_IMPORTS = {"gplearn", "PySR", "eureqa", "deap", "pysr"}
SOFT_PENALTY_FORBIDDEN = 0.30


# ── Utilities ────────────────────────────────────────────────────────────────

def _load_solution():
    """Import solution module from /solution/discover.py"""
    path = os.path.join(SOLUTION_DIR, "discover.py")
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        spec = importlib.util.spec_from_file_location("discover", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, f"Import error: {e}"


def _count_ast_nodes(expr_str: str) -> int:
    """Count AST nodes in expression string."""
    try:
        tree = ast.parse(expr_str, mode='eval')
        return sum(1 for _ in ast.walk(tree))
    except Exception:
        return 9999


def _count_complexity(expr_str: str) -> int:
    """MDL-like complexity: n_ops + 2 * n_constants"""
    try:
        tree = ast.parse(expr_str, mode='eval')
        n_ops = sum(1 for node in ast.walk(tree)
                    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)))
        n_constants = sum(1 for node in ast.walk(tree)
                          if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)))
        return n_ops + 2 * n_constants
    except Exception:
        return 9999


def _eval_expression(expr_str: str, X: np.ndarray, n_vars: int) -> Optional[np.ndarray]:
    """Safely evaluate expression on input array X (N, n_vars)."""
    # pykan's auto_symbolic may emit bare names like tanh, sqrt, log, abs
    # — extend namespace so these resolve correctly
    ns = {"np": np, "math": math,
          "tanh": np.tanh, "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
          "sin": np.sin, "cos": np.cos, "abs": np.abs, "pi": np.pi}
    for i in range(1, n_vars + 1):
        ns[f"x{i}"] = X[:, i - 1]
    try:
        result = eval(expr_str, ns)
        if isinstance(result, (int, float)):
            result = np.full(len(X), float(result))
        result = np.array(result, dtype=np.float64)
        return result
    except Exception:
        return None


def _safe_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE. Returns 1.0 (worst) on failure."""
    if y_pred is None:
        return 1.0
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return 1.0
    yt, yp = y_true[mask], y_pred[mask]
    # y-normalization protection
    y_std = np.std(yt)
    if y_std < 1e-10:
        return 1.0
    # Clip extreme predictions
    yp_clipped = np.clip(yp, yt.mean() - 100 * y_std, yt.mean() + 100 * y_std)
    rmse = np.sqrt(np.mean((yt - yp_clipped) ** 2))
    return float(rmse / y_std)


def _safe_medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error, clipped at 1.0."""
    if y_pred is None:
        return 1.0
    mask = np.isfinite(y_pred) & np.isfinite(y_true) & (np.abs(y_true) > 1e-10)
    if mask.sum() < 10:
        return 1.0
    yt, yp = y_true[mask], y_pred[mask]
    mape = float(np.median(np.abs((yt - yp) / np.abs(yt))))
    return min(1.0, mape)


def _build_ood_samples(domains: dict, n_vars: int, regime: str, n_samples: int, rng) -> np.ndarray:
    """
    Build OOD samples safely within domain constraints.
    regime: 'iid', 'ood1' (symmetric +/-30%), 'ood2' (x1 2x, others nominal)
    """
    X = np.zeros((n_samples, n_vars))
    for i in range(1, n_vars + 1):
        key = f"x{i}"
        lo, hi = domains[key]
        span = hi - lo

        if regime == "iid":
            X[:, i - 1] = rng.uniform(lo, hi, n_samples)
        elif regime == "ood1":
            ext = 0.30 * span
            new_lo = max(lo - ext, lo * 0.5 if lo > 0 else lo - ext)
            new_hi = hi + ext
            X[:, i - 1] = rng.uniform(new_lo, new_hi, n_samples)
        elif regime == "ood2":
            if i == 1:
                new_hi = hi + span  # 2x range for x1
                X[:, i - 1] = rng.uniform(lo, new_hi, n_samples)
            else:
                X[:, i - 1] = rng.uniform(lo, hi, n_samples)
    return X


def _generate_ground_truth_y(dataset_id: str, X: np.ndarray, gt: dict) -> np.ndarray:
    """Generate ground truth y values using known law + constant k."""
    k = gt[dataset_id]["k"]
    x1, x2 = X[:, 0], X[:, 1]
    if dataset_id == "A":
        return k * x1 * x2
    elif dataset_id == "B":
        return k * np.sin(x1) + x2 ** 2
    elif dataset_id == "C":
        return k * x1 / (x1 + x2)
    elif dataset_id == "D":
        return k * np.exp(-x1) * x2
    else:
        raise ValueError(f"Unknown dataset: {dataset_id}")


def _check_forbidden_imports(solution_path: str) -> bool:
    """Check for forbidden symbolic regression libraries in direct imports."""
    try:
        with open(solution_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in FORBIDDEN_IMPORTS:
                        return True
        return False
    except Exception:
        return False


# ── Judge steps ──────────────────────────────────────────────────────────────

class JudgeResult:
    def __init__(self):
        self.steps = []
        self.dataset_scores = {}
        self.dataset_details = {}   # per-dataset metrics for RL feedback
        self.forbidden_penalty = 0.0
        self.final_score = 0.0
        self.passed = False

    def log(self, step: str, status: str, detail: str = ""):
        icon = "✅" if status == "ok" else ("⚠️" if status == "warn" else "❌")
        msg = f"{icon} [{step}] {detail}"
        self.steps.append(msg)
        print(msg)

    def summary(self) -> dict:
        return {
            "final_score": round(self.final_score, 4),
            "passed": bool(self.passed),
            "dataset_scores": {k: round(float(v), 4) for k, v in self.dataset_scores.items()},
            "dataset_details": self.dataset_details,
            "forbidden_penalty": self.forbidden_penalty,
            "steps": self.steps,
        }


def run_judge() -> dict:
    result = JudgeResult()

    # ── Load ground truth (judge only) ───────────────────────────────────────
    if not os.path.exists(GROUND_TRUTH_PATH):
        result.log("INIT", "fail", f"ground_truth.json not found at {GROUND_TRUTH_PATH}")
        return result.summary()
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    # ── Load domains ─────────────────────────────────────────────────────────
    domains_path = os.path.join(DATA_DIR, "domains.json")
    with open(domains_path) as f:
        all_domains = json.load(f)

    # ── STEP 1: File & interface check ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("JUDGE — KAN Symbolic Regression RL Environment")
    print("=" * 60)

    solution_path = os.path.join(SOLUTION_DIR, "discover.py")
    if not os.path.exists(solution_path):
        result.log("Step1", "fail", f"discover.py not found at {solution_path}")
        result.final_score = 0.0
        return result.summary()

    mod, err = _load_solution()
    if mod is None:
        result.log("Step1", "fail", f"Import failed: {err}")
        result.final_score = 0.0
        return result.summary()

    if not hasattr(mod, "discover_law") or not hasattr(mod, "predict"):
        result.log("Step1", "fail", "Missing discover_law() or predict() functions")
        result.final_score = 0.0
        return result.summary()

    result.log("Step1", "ok", "discover.py loaded, discover_law() and predict() found")

    # Smoke test
    try:
        smoke_df = pd.read_csv(os.path.join(DATA_DIR, "train_A.csv"))
        X_smoke = smoke_df[["x1", "x2"]].values[:5]
        expr_smoke = mod.discover_law("A")
        pred_smoke = mod.predict("A", X_smoke)
        assert isinstance(expr_smoke, str), "discover_law must return str"
        assert hasattr(pred_smoke, "__len__"), "predict must return array-like"
        result.log("Step1", "ok", f"Smoke test passed. expr='{ expr_smoke[:60]}...'")
    except Exception as e:
        result.log("Step1", "fail", f"Smoke test failed: {e}")
        result.final_score = 0.0
        return result.summary()

    # ── STEP 2: Import audit (soft penalty) ──────────────────────────────────
    if _check_forbidden_imports(solution_path):
        result.forbidden_penalty = SOFT_PENALTY_FORBIDDEN
        result.log("Step2", "warn",
                   f"Forbidden symbolic regression library detected. "
                   f"Soft penalty: -{SOFT_PENALTY_FORBIDDEN}")
    else:
        result.log("Step2", "ok", "No forbidden imports detected")

    # ── STEP 2b: KAN conformity via traces ───────────────────────────────────
    kan_conformity_scores = {}
    for did in DATASET_IDS:
        trace_path = os.path.join(TRACE_DIR, f"trace_{did}.json")
        if not os.path.exists(trace_path):
            result.log("Step2b", "fail", f"No trace for dataset {did} — train_kan() not called")
            kan_conformity_scores[did] = 0.0
            continue
        with open(trace_path) as f:
            trace = json.load(f)

        checks = []
        # 1. Loss improvement (robust criterion)
        lh = trace.get("loss_history", [])
        if len(lh) >= 10:
            initial = lh[0]
            final = lh[-1]
            improvement = (initial - final) / (initial + 1e-10)
            # Moving average slope check
            window = max(5, len(lh) // 10)
            ma = np.convolve(lh, np.ones(window) / window, mode='valid')
            slope_neg = np.sum(np.diff(ma) < 0) / max(len(np.diff(ma)), 1)
            loss_ok = (final < 0.70 * initial) or (slope_neg >= 0.70)
            checks.append(("loss_improvement", loss_ok,
                           f"final/initial={final:.4f}/{initial:.4f}, neg_slope={slope_neg:.2f}"))
        else:
            checks.append(("loss_history", False, f"Only {len(lh)} steps recorded"))

        # 2. KAN submodules
        checks.append(("kan_verified", trace.get("kan_verified", False),
                       "KAN submodules detected" if trace.get("kan_verified") else "No KAN submodules"))

        # 3. Param count sanity
        n_params = trace.get("param_count", 0)
        checks.append(("param_count", n_params > 0, f"n_params={n_params}"))

        # Score
        n_ok = sum(1 for _, ok, _ in checks if ok)
        score = n_ok / len(checks)
        kan_conformity_scores[did] = score

        for check_name, ok, detail in checks:
            result.log(f"Step2b[{did}]", "ok" if ok else "warn",
                       f"{check_name}: {detail}")

    # ── STEP 3-7: Per-dataset scoring ────────────────────────────────────────
    rng = np.random.default_rng(999)  # Fixed seed for reproducibility
    N_TEST = 200  # toy: smaller than 300

    for did in DATASET_IDS:
        print(f"\n--- Dataset {did} ---")
        domains = all_domains[did]
        n_vars = gt[did]["n_vars"]
        ast_cap = AST_CAPS.get(n_vars, 60)

        try:
            expr = mod.discover_law(did)
        except Exception as e:
            result.log(f"Step3[{did}]", "fail", f"discover_law raised: {e}")
            result.dataset_scores[did] = 0.0
            continue

        # ── Step 3: Expression validity + complexity ──────────────────────────
        n_nodes = _count_ast_nodes(expr)
        complexity = _count_complexity(expr)
        result.log(f"Step3[{did}]", "ok" if n_nodes <= ast_cap else "warn",
                   f"AST nodes={n_nodes} (cap={ast_cap}), complexity={complexity}, expr='{expr[:80]}'")

        parsimony_score = 0.0 if n_nodes > ast_cap else math.exp(-0.05 * complexity)
        parsimony_score = max(0.0, min(1.0, parsimony_score))

        # Test eval on in-domain points
        X_indom = _build_ood_samples(domains, n_vars, "iid", 20, rng)
        y_test = _eval_expression(expr, X_indom, n_vars)
        if y_test is None or not np.all(np.isfinite(y_test)):
            result.log(f"Step3[{did}]", "fail", "Expression produces NaN/Inf on in-domain points")
            result.dataset_scores[did] = 0.0
            continue

        result.log(f"Step3[{did}]", "ok", "Expression evaluates cleanly on in-domain points")

        # ── Step 4: Consistency expr ↔ KAN ───────────────────────────────────
        try:
            X_cons = _build_ood_samples(domains, n_vars, "iid", 400, rng)
            y_expr_cons = _eval_expression(expr, X_cons, n_vars)
            X_cons_df = pd.DataFrame(X_cons, columns=[f"x{i}" for i in range(1, n_vars + 1)])
            y_kan_cons = np.array(mod.predict(did, X_cons), dtype=np.float64).flatten()

            if y_expr_cons is not None and len(y_kan_cons) == len(y_expr_cons):
                eps = np.std(y_kan_cons) * 0.01 + 1e-10
                divergence = float(np.median(np.abs(y_expr_cons - y_kan_cons) / (np.abs(y_kan_cons) + eps)))
                if divergence < 0.05:
                    consistency_score = 1.0
                elif divergence < 0.20:
                    consistency_score = math.exp(-10 * divergence)
                else:
                    consistency_score = 0.0
                result.log(f"Step4[{did}]", "ok" if consistency_score > 0.5 else "warn",
                           f"divergence={divergence:.4f}, consistency={consistency_score:.3f}")
            else:
                consistency_score = 0.0
                result.log(f"Step4[{did}]", "warn", "Could not compute consistency")
        except Exception as e:
            consistency_score = 0.0
            result.log(f"Step4[{did}]", "warn", f"Consistency check failed: {e}")

        generalization_halved = consistency_score == 0.0

        # ── Step 5: Multi-OOD generalization ─────────────────────────────────
        regime_scores = {}
        for regime, weight in [("iid", 0.40), ("ood1", 0.35), ("ood2", 0.25)]:
            X_reg = _build_ood_samples(domains, n_vars, regime, N_TEST, rng)
            y_true = _generate_ground_truth_y(did, X_reg, gt)
            y_pred = _eval_expression(expr, X_reg, n_vars)

            # Overflow protection
            if y_pred is not None:
                y_std = np.std(y_true)
                if y_std > 0:
                    overflow_frac = np.mean(np.abs(y_pred) > 100 * y_std)
                    if overflow_frac > 0.10:
                        result.log(f"Step5[{did}][{regime}]", "warn",
                                   f"Overflow: {overflow_frac:.1%} of predictions unstable → score=0")
                        regime_scores[regime] = 0.0
                        continue

            nrmse = _safe_nrmse(y_true, y_pred)
            medape = _safe_medape(y_true, y_pred)
            score = max(0.0, 1 - nrmse) * max(0.0, 1 - medape)
            regime_scores[regime] = score
            result.log(f"Step5[{did}][{regime}]", "ok",
                       f"NRMSE={nrmse:.4f}, MedAPE={medape:.4f}, score={score:.4f}")

        gen_score = (
            0.40 * regime_scores.get("iid", 0.0) +
            0.35 * regime_scores.get("ood1", 0.0) +
            0.25 * regime_scores.get("ood2", 0.0)
        )
        if generalization_halved:
            gen_score *= 0.5
            result.log(f"Step5[{did}]", "warn",
                       f"Generalization halved due to consistency=0. gen={gen_score:.4f}")

        # ── Per-dataset final score ───────────────────────────────────────────
        kan_conf = kan_conformity_scores.get(did, 0.0)
        ds_score = (
            W_GENERALIZATION * gen_score +
            W_PARSIMONY * parsimony_score +
            W_CONSISTENCY * consistency_score +
            W_KAN_CONFORMITY * kan_conf
        )
        result.dataset_details[did] = {
            "score":           round(ds_score, 4),
            "generalization":  round(gen_score, 4),
            "parsimony":       round(parsimony_score, 4),
            "consistency":     round(consistency_score, 4),
            "kan_conformity":  round(kan_conf, 4),
            "expression":      expr if 'expr' in dir() else "",
            # Per-regime scores for feedback diagnosis
            "iid_score":       round(regime_scores.get("iid",  0.0), 4),
            "ood1_score":      round(regime_scores.get("ood1", 0.0), 4),
            "ood2_score":      round(regime_scores.get("ood2", 0.0), 4),
            "ast_nodes":       n_nodes,
            "ast_cap":         ast_cap,
        }
        result.dataset_scores[did] = round(ds_score, 4)
        result.log(f"Score[{did}]", "ok",
                   f"gen={gen_score:.3f} | parsimony={parsimony_score:.3f} | "
                   f"consistency={consistency_score:.3f} | kan_conf={kan_conf:.3f} "
                   f"→ DATASET SCORE = {ds_score:.4f}")

    # ── Final score ───────────────────────────────────────────────────────────
    if result.dataset_scores:
        raw = np.mean(list(result.dataset_scores.values()))
    else:
        raw = 0.0

    result.final_score = max(0.0, raw - result.forbidden_penalty)
    result.passed = result.final_score >= PASS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"FINAL SCORE: {result.final_score:.4f}  ({'PASS ✅' if result.passed else 'FAIL ❌'})")
    print(f"Threshold: {PASS_THRESHOLD}")
    for did, s in result.dataset_scores.items():
        print(f"  Dataset {did}: {s:.4f}")
    print("=" * 60)

    return result.summary()


if __name__ == "__main__":
    summary = run_judge()
    out_path = os.path.join(SOLUTION_DIR, "judge_result.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    sys.exit(0 if summary["passed"] else 1)