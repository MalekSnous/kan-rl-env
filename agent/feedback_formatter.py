"""
agent/feedback_formatter.py
Converts judge_result.json → rl_feedback.json after each RL round.
Called by run.sh between agent and next round.
"""

import json
import os
import sys

SOLUTION_DIR = os.environ.get("SOLUTION_DIR", "solution")
TRACE_DIR    = os.environ.get("TRACE_DIR",    "/tmp/kan_traces")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_lib_from_expr(expr: str) -> set:
    """Detect which function families are present in the expression."""
    present = set()
    if "np.sin" in expr or "sin(" in expr:
        present.add("sin")
    if "np.cos" in expr or "cos(" in expr:
        present.add("cos")
    if "np.exp" in expr or "exp(" in expr:
        present.add("exp")
    if "np.sqrt" in expr or "sqrt(" in expr:
        present.add("sqrt")
    if "np.log" in expr or "log(" in expr:
        present.add("log")
    if "**2" in expr or "x^2" in expr:
        present.add("x^2")
    return present


PARAM_TO_WIDTH = {
    84:  "[2,1,1]",
    156: "[2,2,1]",
    220: "[2,3,1]",
    284: "[2,4,1]",
    348: "[2,5,1]",
}

def _infer_width(param_count: int) -> str:
    if not param_count:
        return "unknown"
    if param_count in PARAM_TO_WIDTH:
        return PARAM_TO_WIDTH[param_count]
    closest = min(PARAM_TO_WIDTH.keys(), key=lambda x: abs(x - param_count))
    return PARAM_TO_WIDTH[closest] + "~"


def _is_stale_model(did: str, trace: dict, history: list) -> bool:
    """
    Detect stale model: same (param_count, final_loss) as a previous round.
    initial_loss alone caused false positives — same arch = same deterministic init.
    """
    if not trace or not history:
        return False
    cur_params = trace.get("param_count")
    cur_final  = trace.get("final_loss")
    if cur_params is None or cur_final is None:
        return False
    for past_round in history[-4:]:
        past_ds = past_round.get("datasets", {}).get(did, {})
        p_params = past_ds.get("param_count")
        p_final  = past_ds.get("final_loss")
        if (p_params is not None and p_final is not None
                and p_params == cur_params
                and abs(cur_final - p_final) < 1e-6):
            return True
    return False


def _best_config(did: str, history: list) -> dict | None:
    """Return the config (width, score) from the best scoring round for this dataset."""
    best_score = 0.0
    best = None
    for past_round in history:
        past_ds = past_round.get("datasets", {}).get(did, {})
        s = past_ds.get("score", 0)
        if s > best_score:
            best_score = s
            cfg = past_ds.get("config", {})
            best = {
                "score":       round(s, 4),
                "width":       _infer_width(past_ds.get("param_count", 0)),
                "param_count": past_ds.get("param_count", 0),
                "round":       past_round.get("round", "?"),
                "lr":          cfg.get("lr", "?"),
                "steps":       cfg.get("steps", "?"),
                "lambda_reg":  cfg.get("lambda_reg", "?"),
            }
    return best


# ── Core advice logic ─────────────────────────────────────────────────────────

def _advise(did: str, ds: dict, trace: dict, history: list) -> list[str]:
    """
    Generate targeted advice for one dataset.
    Priority: correctness > extrapolation > complexity.
    """
    advice = []

    gen         = ds.get("generalization", 0)
    parsimony   = ds.get("parsimony", 0)
    consistency = ds.get("consistency", 0)
    kan_conf    = ds.get("kan_conformity", 0)
    iid         = ds.get("iid_score",  0)
    ood1        = ds.get("ood1_score", 0)
    ood2        = ds.get("ood2_score", 0)
    ast_nodes   = ds.get("ast_nodes",  0)
    ast_cap     = ds.get("ast_cap", 60)
    expression  = ds.get("expression", "")

    final_loss   = trace.get("final_loss")   if trace else None
    initial_loss = trace.get("initial_loss") if trace else None

    # ── 1. KAN not trained at all ─────────────────────────────────────────────
    if kan_conf == 0:
        advice.append(
            "train_kan() crashed or was not called — "
            "verify width=[2,X,1] (first dim MUST be 2, datasets have 2 input features)"
        )
        return advice

    # ── 2. Stale model (reused from previous round) ───────────────────────────
    if _is_stale_model(did, trace, history):
        advice.append(
            "STALE MODEL detected — same initial_loss as a previous round. "
            "The model was NOT retrained. Always retrain all 4 datasets each round."
        )

    # ── 3. Constant expression ────────────────────────────────────────────────
    is_constant = bool(expression) and all(
        c.isdigit() or c in ".-e " for c in expression.replace("np.", "").replace(" ", "")
    )
    if is_constant or (consistency < 0.5 and iid < 0.1):
        advice.append(
            f"CONSTANT/TRIVIAL EXPRESSION '{expression[:40]}' — KAN did not converge. "
            "Fix: increase steps to 800+, use lr=0.005, "
            "call auto_symbolic AFTER training not before."
        )
        return advice

    # ── 4. Loss diagnosis ─────────────────────────────────────────────────────
    if final_loss is not None and initial_loss is not None and initial_loss > 1e-8:
        ratio = final_loss / initial_loss
        overfit_gap = iid - ood2

        if ratio > 0.80:
            # Loss barely reduced — underfitting, but check overfit too
            if overfit_gap > 0.4:
                # Already overfitting despite high loss — regularise, not more steps
                advice.append(
                    f"loss barely reduced ({ratio:.0%} of initial: {initial_loss:.4f}→{final_loss:.4f}) "
                    "AND overfit detected — increase lambda_reg to 0.01, reduce steps to 300"
                )
            else:
                advice.append(
                    f"loss barely reduced ({ratio:.0%} of initial: {initial_loss:.4f}→{final_loss:.4f}) "
                    "— increase steps to 600+, try lr=0.005"
                )
        elif ratio > 0.50:
            if overfit_gap > 0.4:
                advice.append(
                    f"loss partially reduced ({ratio:.0%} of initial: {initial_loss:.4f}→{final_loss:.4f}) "
                    "but overfit detected (iid-ood2={:.2f}) — "
                    "increase lambda_reg to 0.01 instead of more steps".format(overfit_gap)
                )
            else:
                advice.append(
                    f"loss partially reduced ({ratio:.0%} of initial: {initial_loss:.4f}→{final_loss:.4f}) "
                    "— increase steps to 500+"
                )
        elif final_loss > 0.15:
            advice.append(
                f"loss still high ({final_loss:.4f}) — "
                "try refine(5) then retrain 300 steps at lr=0.005"
            )
        elif ratio < 0 or (final_loss > initial_loss):
            advice.append(
                f"DIVERGED: loss increased {final_loss/initial_loss*100:.0f}% "
                f"({initial_loss:.4f}→{final_loss:.4f}) — "
                "MANDATORY: reduce lr by 10x on BOTH steps. Do NOT increase width or steps."
            )

    # ── 5. OOD2 / extrapolation diagnosis (most important) ───────────────────
    overfit_gap = iid - ood2
    lib_present = _infer_lib_from_expr(expression)

    if ood2 < 0.10:
        # Critically low OOD2 — diagnose WHY
        if overfit_gap > 0.5:
            # Clear overfit — more regularisation needed
            advice.append(
                "OOD2 critically low (expression does NOT extrapolate). "
                "Cause: overfit. Fix: increase lambda_reg to 0.01-0.05, "
                "reduce steps to 200-300, use [2,2,1] architecture."
            )
        elif not lib_present.intersection({"exp", "sqrt"}):
            # Lib too restrictive for the underlying law
            advice.append(
                "OOD2 critically low AND lib is restrictive — "
                "the true law likely needs 'exp' or 'sqrt'. "
                "Try lib=['x','x^2','sin','exp','sqrt'] — do NOT reduce lib further."
            )
        else:
            advice.append(
                "OOD2 critically low — expression structure may be wrong. "
                "Try different lib: if exp present, try removing it; "
                "if only sin/x^2, add exp or sqrt."
            )
    elif ood2 < 0.25:
        if overfit_gap > 0.4:
            advice.append(
                f"OOD2 low (ood2={ood2:.2f}, iid-ood2={overfit_gap:.2f}). "
                "Increase lambda_reg to 0.005-0.01 to force simpler expression."
            )
        else:
            advice.append(
                f"OOD2 moderate (ood2={ood2:.2f}) — expression partially extrapolates. "
                "Try adding 'exp' or 'sqrt' to lib if not already present."
            )
    elif ood2 >= 0.25 and overfit_gap > 0.5:
        advice.append(
            f"OOD2 acceptable (ood2={ood2:.2f}) but large IID-OOD2 gap ({overfit_gap:.2f}). "
            "Slight overfit — try lambda_reg=0.005 to reduce gap."
        )

    # ── 6. Complexity / parsimony ─────────────────────────────────────────────
    if ast_nodes > ast_cap:
        excess = ast_nodes / ast_cap
        advice.append(
            f"PARSIMONY=0 ({ast_nodes} nodes, cap={ast_cap}, x{excess:.1f}) — losing 0.20 score weight. "
            "Reduce lib to 2-3 functions based on what's in the current expression "
            "(keep only function families already present with high r2)."
        )
    elif ast_nodes > ast_cap * 0.8:
        advice.append(
            f"expression near complexity cap ({ast_nodes}/{ast_cap} nodes) — parsimony at risk. "
            "Remove one function from lib."
        )

    # ── 7. Missing variables ──────────────────────────────────────────────────
    if "x1" not in expression and gen < 0.5:
        advice.append("x1 absent from expression — KAN ignores first feature. Retrain.")
    if "x2" not in expression and gen < 0.5:
        advice.append("x2 absent from expression — KAN ignores second feature. Retrain.")

    # ── 8. Good result ────────────────────────────────────────────────────────
    if not advice:
        if ast_nodes > 40:
            advice.append(
                f"good result — try smaller lib ['x','x^2','sin'] "
                f"to reduce AST nodes further ({ast_nodes} → target <40)"
            )
        else:
            advice.append("good result — stable, try refine(5) for marginal improvement")

    return advice[:3]  # cap at 3 to save context tokens


def _priority(score: float) -> str:
    if score < 0.30:
        return "🔴 HIGH PRIORITY"
    elif score < 0.55:
        return "🟡 NEEDS IMPROVEMENT"
    else:
        return "🟢 ACCEPTABLE"


# ── Global advice ─────────────────────────────────────────────────────────────

def _global_advice(score: float, dfs: dict, order: list, history: list) -> list[str]:
    advice = []

    # Near-threshold signal: identify frozen and focus datasets
    threshold = 0.65
    gap = threshold - score
    frozen = [did for did in "ABCD" if dfs[did]["score"] >= 0.65]
    weak   = [did for did in order if dfs[did]["score"] < 0.65]

    if 0 < gap <= 0.05 and frozen:
        frozen_str = ", ".join(frozen)
        focus_str  = ", ".join(weak[:2])
        advice.append(
            f"VERY CLOSE: score={score:.4f}, need {gap:.4f} more to PASS. "
            f"FREEZE datasets {frozen_str} — do NOT change their config. "
            f"Focus ALL effort on {focus_str} only."
        )
    elif frozen:
        frozen_str = ", ".join(frozen)
        advice.append(
            f"Datasets {frozen_str} are above threshold — FREEZE their config (same width/steps/lambda_reg). "
            "Only modify the weakest dataset each round."
        )

    # Detect plateau (no improvement in last 3 rounds)
    if len(history) >= 4:
        recent = [h.get("final_score", 0) for h in history[-4:]]
        if max(recent) - min(recent) < 0.02:
            advice.append(
                "PLATEAU detected — score has not improved in 4 rounds. "
                "Try a fundamentally different strategy: change ALL hyperparameters, "
                "or try lambda_reg=0.05 with steps=200 for aggressive simplification."
            )

    # Overfit dominance
    n_overfit = sum(
        1 for did, df in dfs.items()
        if df.get("metrics", {}).get("iid_score", 0) - df.get("metrics", {}).get("ood2_score", 0) > 0.5
    )
    if n_overfit >= 3:
        advice.append(
            f"{n_overfit}/4 datasets are overfitting. "
            "Global fix: set lambda_reg=0.01 for ALL datasets. "
            "This is more important than tuning steps or architecture."
        )

    if score < 0.30:
        advice.append(
            f"Very low score — focus on {order[0]} first. "
            "Verify train_kan() is called for ALL 4 datasets."
        )

    return advice


# ── Main formatter ────────────────────────────────────────────────────────────

def format_feedback(rl_round: int, judge_path: str, history_path: str) -> dict:
    if not os.path.exists(judge_path):
        print(f"[feedback] No judge result at {judge_path}")
        sys.exit(1)

    result      = json.load(open(judge_path))
    final_score = result.get("final_score", 0)
    details     = result.get("dataset_details", {})
    flat_scores = result.get("dataset_scores", {})

    datasets = {}
    for did in "ABCD":
        if did in details:
            datasets[did] = details[did]
        elif did in flat_scores:
            datasets[did] = {"score": flat_scores[did]}

    # Load history
    history = []
    if os.path.exists(history_path):
        history = json.load(open(history_path))

    # Load traces
    traces = {}
    for did in "ABCD":
        tp = os.path.join(TRACE_DIR, f"trace_{did}.json")
        traces[did] = json.load(open(tp)) if os.path.exists(tp) else {}

    # Track best per dataset
    best_scores = {did: 0.0 for did in "ABCD"}
    best_exprs  = {}
    for past in history:
        for did, ds in past.get("datasets", {}).items():
            if ds.get("score", 0) > best_scores[did]:
                best_scores[did] = ds["score"]
                best_exprs[did]  = ds.get("expression", "?")

    # Build per-dataset feedback
    dataset_feedback = {}
    for did in "ABCD":
        ds    = datasets.get(did, {})
        score = ds.get("score", 0)
        expr  = ds.get("expression", "?")
        prev  = best_scores.get(did, 0)
        delta = score - prev

        advice = _advise(did, ds, traces.get(did, {}), history)

        dataset_feedback[did] = {
            "score":    round(score, 4),
            "delta":    round(delta, 4),
            "priority": _priority(score),
            "expression": expr,
            "best_expression_so_far": best_exprs.get(did, expr),
            "best_config": _best_config(did, history),
            "metrics": {
                "generalization": round(ds.get("generalization", 0), 4),
                "parsimony":      round(ds.get("parsimony", 0), 4),
                "consistency":    round(ds.get("consistency", 0), 4),
                "kan_conformity": round(ds.get("kan_conformity", 0), 4),
                "iid_score":      round(ds.get("iid_score",  0), 4),
                "ood1_score":     round(ds.get("ood1_score", 0), 4),
                "ood2_score":     round(ds.get("ood2_score", 0), 4),
                "ast_nodes":      ds.get("ast_nodes", 0),
            },
            "advice": advice,
        }

    priority_order = sorted("ABCD", key=lambda d: dataset_feedback[d]["score"])

    feedback = {
        "rl_round":       rl_round,
        "final_score":    round(final_score, 4),
        "passed":         final_score >= 0.65,
        "target_score":   0.65,
        "priority_order": priority_order,
        "datasets":       dataset_feedback,
        "history_scores": [round(h.get("final_score", 0), 4) for h in history],
        "global_advice":  _global_advice(final_score, dataset_feedback, priority_order, history),
    }

    # Append to history
    history.append({
        "round":       rl_round,
        "final_score": final_score,
        "datasets": {
            did: {
                "score":       dataset_feedback[did]["score"],
                "expression":  dataset_feedback[did]["expression"],
                "param_count": traces.get(did, {}).get("param_count"),
                "final_loss":  traces.get(did, {}).get("final_loss"),
                "config":      traces.get(did, {}).get("config", {}),
            }
            for did in "ABCD"
        },
    })
    json.dump(history, open(history_path, "w"), indent=2)
    feedback["history_scores"] = [round(h.get("final_score", 0), 4) for h in history]

    return feedback


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    rl_round      = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    judge_path    = os.path.join(SOLUTION_DIR, "judge_result.json")
    feedback_path = os.path.join(SOLUTION_DIR, "rl_feedback.json")
    history_path  = os.path.join(SOLUTION_DIR, "rl_history.json")

    feedback = format_feedback(rl_round, judge_path, history_path)
    json.dump(feedback, open(feedback_path, "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"RL ROUND {rl_round} — Score: {feedback['final_score']} "
          f"({'PASS ✅' if feedback['passed'] else 'FAIL ❌'})")
    print(f"Priority order: {' > '.join(feedback['priority_order'])}")
    for did in feedback['priority_order']:
        df = feedback['datasets'][did]
        m  = df['metrics']
        delta_str = f"({df['delta']:+.3f})" if df['delta'] != 0 else ""
        bc = df.get("best_config")
        if bc:
            bc_cfg = f"lr={bc.get('lr','?')} steps={bc.get('steps','?')} lambda={bc.get('lambda_reg','?')}"
            bc_str = f" [best: {bc['width']} r{bc['round']} score={bc['score']} | {bc_cfg}]"
        else:
            bc_str = ""
        print(f"  {df['priority']} {did}: {df['score']} {delta_str}{bc_str}"
              f"  — {df['advice'][0]}")
    print(f"{'='*60}")
    print(f"Feedback written → {feedback_path}")