"""
env_api/kan_env.py
Instrumented KAN training API. The agent MUST use train_kan().
The judge reads the trace directly — cannot be spoofed via log files.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

# Force non-interactive matplotlib before any pykan import
import matplotlib
matplotlib.use('Agg')

TRACE_DIR = os.environ.get("TRACE_DIR", "/tmp/kan_traces")
os.makedirs(TRACE_DIR, exist_ok=True)


def _model_hash(model) -> str:
    params = [p.detach().cpu().numpy().tobytes() for p in model.parameters()]
    return hashlib.sha256(b"".join(params)).hexdigest()[:16]


def _check_kan_submodules(model) -> bool:
    model_str = str(type(model)).lower()
    module_names = [type(m).__name__.lower() for m in model.modules()]
    indicators = ["kan", "spline", "bspline"]
    return any(
        ind in name
        for ind in indicators
        for name in [model_str] + module_names
    )


def train_kan(dataset_id: str, model, config: dict) -> tuple:
    """
    Instrumented KAN training. Required by the judge.

    Args:
        dataset_id : "A" | "B" | "C" | "D"
        model      : KAN instance from pykan
        config     : dict with lr, steps, lambda_reg (optional), seed (optional)

    Returns:
        (trained_model, trace)
    """
    assert dataset_id in ("A", "B", "C", "D"), f"Unknown dataset_id: {dataset_id}"

    data_path = os.environ.get("DATA_DIR", "data")
    csv = os.path.join(data_path, f"train_{dataset_id}.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Dataset not found: {csv}")

    import pandas as pd
    df = pd.read_csv(csv)
    n_vars = len(df.columns) - 1
    X = torch.tensor(df[[f"x{i}" for i in range(1, n_vars + 1)]].values, dtype=torch.float32)
    y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    lr       = float(config.get("lr", 0.01))
    steps    = int(config.get("steps", 100))
    seed     = int(config.get("seed", 42))
    lamb     = float(config.get("lambda_reg", 0.001))

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_dict = {
        "train_input": X, "train_label": y,
        "test_input":  X, "test_label":  y,
    }

    loss_history = []
    t0 = time.time()

    try:
        # Exact signature (inspected from container):
        # fit(self, dataset, opt='LBFGS', steps=100, log=1, lamb=0.0,
        #     lamb_l1=1.0, lamb_entropy=2.0, lr=1.0, batch=-1,
        #     update_grid=True, device='cpu', ...)
        results = model.fit(
            dataset_dict,
            opt="Adam",     # Adam converges faster than LBFGS on small datasets
            steps=steps,
            log=steps,      # print only once at the end (log every N steps)
            lamb=lamb,
            lr=lr,
            update_grid=True,
            device="cpu",
        )
        # results is a dict with keys like 'train_loss', 'test_loss'
        if isinstance(results, dict):
            key = next((k for k in ("train_loss", "loss", "train_losses") if k in results), None)
            loss_history = [float(x) for x in results[key]] if key else []
        else:
            loss_history = []

        # Fallback: compute final loss manually if history empty
        if not loss_history:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                loss_val = float(torch.nn.functional.mse_loss(pred, y).item())
            loss_history = [loss_val]

    except Exception as e:
        raise RuntimeError(f"KAN training failed for dataset {dataset_id}: {e}")

    duration = time.time() - t0

    trace = {
        "dataset_id":   dataset_id,
        "loss_history": loss_history,
        "n_steps":      len(loss_history),
        "initial_loss": loss_history[0]  if loss_history else None,
        "final_loss":   loss_history[-1] if loss_history else None,
        "lr":           lr,
        "param_count":  sum(p.numel() for p in model.parameters()),
        "model_hash":   _model_hash(model),
        "config":       config,
        "duration_s":   round(duration, 2),
        "kan_verified": _check_kan_submodules(model),
        "timestamp":    time.time(),
    }

    trace_path = os.path.join(TRACE_DIR, f"trace_{dataset_id}.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"[env_api] {dataset_id} | steps={steps} | "
          f"loss {trace['initial_loss']:.4f}→{trace['final_loss']:.4f} | "
          f"hash={trace['model_hash']} | {duration:.1f}s")

    return model, trace


def get_trace(dataset_id: str) -> dict:
    p = os.path.join(TRACE_DIR, f"trace_{dataset_id}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def get_all_traces() -> dict:
    return {did: get_trace(did) for did in ("A", "B", "C", "D")}


def extract_symbolic(model, n_vars: int = 2) -> str:
    """
    Extract a numpy-evaluable symbolic expression from a trained KAN.

    Calls auto_symbolic() then symbolic_formula(), converts the sympy
    expression to a Python/numpy string the agent can use directly in
    discover_law() and eval().

    Args:
        model  : trained KAN instance (after train_kan())
        n_vars : number of input variables (default 2)

    Returns:
        str : numpy-evaluable expression e.g. "2.43 * x1 * x2"
              Falls back to "None" string if extraction fails.

    Example:
        trained_model, trace = train_kan("A", model, config)
        expr = extract_symbolic(trained_model, n_vars=2)
        print(expr)  # "2.4349 * x1 * x2"
    """
    import re

    try:
        # Step 1: fit symbolic functions to each spline activation
        # NOTE: auto_symbolic() switches model to symbolic mode.
        # The caller must save model weights BEFORE calling extract_symbolic()
        # so that predict() (which uses load_state_dict) stays in spline mode.
        model.auto_symbolic()
    except Exception as e:
        print(f"[extract_symbolic] auto_symbolic() failed: {e}")
        return None

    try:
        # Step 2: get the symbolic formula as a sympy expression
        # symbolic_formula() returns ([expr_list], [var_list])
        result = model.symbolic_formula()

        # Handle different return formats across pykan versions
        if isinstance(result, (list, tuple)) and len(result) >= 1:
            formulas = result[0]
            if isinstance(formulas, (list, tuple)) and len(formulas) >= 1:
                expr = formulas[0]
            else:
                expr = formulas
        else:
            expr = result

        expr_str = str(expr)

    except Exception as e:
        print(f"[extract_symbolic] symbolic_formula() failed: {e}")
        return None

    # Step 3: check the formula is not trivially a constant
    # (happens when auto_symbolic fixes all activations to 0 or constant)
    has_variable = bool(re.search(r'x_\d+|x\d+', expr_str))
    if not has_variable:
        print(f"[extract_symbolic] Warning: formula is a constant ({expr_str}). "
              f"KAN may not have converged — consider more steps or lower lr.")

    # Step 4: convert sympy variable names → numpy-compatible names
    # pykan uses x_1, x_2 ... → rename to x1, x2 ...
    expr_str = re.sub(r'x_(\d+)', r'x\1', expr_str)

    # Step 5: convert sympy math → numpy math
    replacements = [
        # trig
        (r'\bsin\b',   'np.sin'),
        (r'\bcos\b',   'np.cos'),
        (r'\btan\b',   'np.tan'),
        (r'\bexp\b',   'np.exp'),
        (r'\blog\b',   'np.log'),
        (r'\bsqrt\b',  'np.sqrt'),
        (r'\babs\b',   'np.abs'),
        (r'\bAbs\b',   'np.abs'),
        # sympy constants
        (r'\bpi\b',    'np.pi'),
        (r'\bE\b',     'np.e'),
        # sympy power operator (already ** in most cases, but just in case)
        (r'\bPow\(([^,]+),\s*([^)]+)\)', r'(\1)**(\2)'),
    ]
    for pattern, replacement in replacements:
        expr_str = re.sub(pattern, replacement, expr_str)

    # Step 6: round long floats to 4 significant figures for readability
    def _round_match(m):
        try:
            val = float(m.group(0))
            # Format to 4 sig figs
            return f"{val:.4g}"
        except Exception:
            return m.group(0)

    expr_str = re.sub(r'-?\d+\.\d{5,}', _round_match, expr_str)

    print(f"[extract_symbolic] → '{expr_str}'")
    return expr_str


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, dataset_id: str) -> str:
    """
    Save a trained KAN model (state_dict + grid metadata).
    Always call BEFORE extract_symbolic().

    Args:
        model      : trained KAN instance
        dataset_id : "A" | "B" | "C" | "D"

    Returns:
        path to saved .pt file
    """
    import json as _json
    solution_dir = os.environ.get("SOLUTION_DIR", "solution")
    models_dir   = os.path.join(solution_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    pt_path   = os.path.join(models_dir, f"{dataset_id}_model.pt")
    meta_path = os.path.join(models_dir, f"{dataset_id}_meta.json")

    # Save weights
    torch.save(model.state_dict(), pt_path)

    # Save grid size — changes after refine(), must match at load time
    grid = int(getattr(model, 'grid', 3))
    _json.dump({"grid": grid, "width": [2, 3, 1], "k": 3}, open(meta_path, "w"))

    print(f"[env_api] Saved {dataset_id} → {pt_path} (grid={grid})")
    return pt_path


def load_model(dataset_id: str):
    """
    Load a saved KAN model and restore symbolic mode.
    Reconstructs the exact same model that was saved (correct grid size).
    Calls auto_symbolic() to match predict() with discover_law().

    Args:
        dataset_id : "A" | "B" | "C" | "D"

    Returns:
        KAN model in eval+symbolic mode
    """
    import json as _json

    solution_dir = os.environ.get("SOLUTION_DIR", "solution")
    models_dir   = os.path.join(solution_dir, "models")
    pt_path      = os.path.join(models_dir, f"{dataset_id}_model.pt")
    meta_path    = os.path.join(models_dir, f"{dataset_id}_meta.json")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"No saved model for dataset {dataset_id}: {pt_path}")

    # Read metadata
    try:
        meta  = _json.load(open(meta_path))
        grid  = meta.get("grid",  3)
        width = meta.get("width", [2, 3, 1])
        k     = meta.get("k",     3)
    except Exception:
        grid, width, k = 3, [2, 3, 1], 3

    # pykan writes history.txt in ./model/ during auto_symbolic
    os.makedirs("./model", exist_ok=True)

    from kan import KAN
    model = KAN(width=width, grid=grid, k=k, seed=42)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))

    # Restore symbolic mode — must match what extract_symbolic() did
    try:
        model.auto_symbolic()
    except Exception as e:
        print(f"[load_model] auto_symbolic() warning for {dataset_id}: {e}")

    model.eval()
    print(f"[env_api] Loaded {dataset_id} ← {pt_path} (grid={grid})")
    return model


# ── Expression-based prediction (garanteed consistency=1.0) ──────────────────

def predict_from_expr(expr_str: str, X) -> 'np.ndarray':
    """
    Evaluate a symbolic expression on input X.
    ALWAYS use this in predict() — guarantees consistency=1.0 with discover_law().

    Args:
        expr_str : numpy-evaluable string e.g. "2.43 * x1 * x2"
        X        : array-like of shape (n, 2)

    Returns:
        np.ndarray of shape (n,)
    """
    X = np.array(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]

    # Evaluate safely
    try:
        result = eval(expr_str, {"np": np, "x1": x1, "x2": x2, "tanh": np.tanh, "sqrt": np.sqrt, "log": np.log,
           "exp": np.exp, "sin": np.sin, "cos": np.cos, "abs": np.abs})
        result = np.array(result, dtype=np.float64).flatten()
    except Exception as e:
        raise ValueError(f"predict_from_expr failed on expr '{expr_str}': {e}")

    # Replace NaN/Inf with 0 to avoid score explosion
    result = np.where(np.isfinite(result), result, 0.0)
    return result