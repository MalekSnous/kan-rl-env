"""
agent/agent.py — LLM Agent using Groq API for KAN symbolic regression.
"""

import os, sys, json, subprocess, time, re
from typing import Optional
import requests

# ── Config ────────────────────────────────────────────────────────────────────
def _require(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise EnvironmentError(f"Missing required env variable: {key}")
    return val

GROQ_API_KEY   = _require("GROQ_API_KEY")
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = _require("GROQ_MODEL")
SOLUTION_DIR   = _require("SOLUTION_DIR", "solution")
DATA_DIR       = _require("DATA_DIR",     "data")
TRACE_DIR      = _require("TRACE_DIR",    "/tmp/kan_traces")
MAX_ITERATIONS = int(_require("MAX_ITERATIONS", "4"))
MAX_TOKENS     = int(_require("MAX_TOKENS",     "2048"))
RL_ROUND       = int(os.environ.get("RL_ROUND", "1"))
FEEDBACK_PATH  = os.path.join(SOLUTION_DIR, "rl_feedback.json")

# ── System prompt — minimal ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
KAN symbolic regression agent. Goal: discover simple expressions for datasets A, B, C, D.

SCORING: score = 0.60*gen + 0.20*parsimony + 0.10*consistency + 0.10*kan_conformity
PARSIMONY: exp(-0.03 * complexity) if ast_nodes <= 80 else 0
  complexity = n_ops + 2*n_constants. Keep expressions under 80 AST nodes.
  Under the cap: simpler is better (10 nodes→0.74, 30 nodes→0.41, 60 nodes→0.16, 79 nodes→0.09).
Short expressions that extrapolate win. Complex ones that memorise lose.

OOD2 DIAGNOSIS (use this when ood2 is low in feedback):
- ood2 < 0.10 AND iid > 0.70 (gap > 0.50): OVERFIT — increase lambda_reg to 0.01, reduce steps
- ood2 < 0.10 AND lib has only sin/x^2: LIB TOO SMALL — add 'exp' or 'sqrt' to lib
- ood2 < 0.25 AND overfit: increase lambda_reg, do NOT increase steps
- Never reduce lib when ood2 is near 0 — that makes extrapolation worse

BEST CONFIG RULE (most important):
- Each dataset has a "best_config" in the feedback (best width + score + lr + steps + lambda_reg).
- If a dataset's score DROPPED vs prev round: revert to its best_config (copy lr/steps/lambda_reg exactly).
- If a dataset's score >= 0.65: FREEZE — do not change config at all. Copy best_config exactly.
- If a dataset's score is 0.60-0.65: keep its config, only tune lambda_reg.
- Only change architecture if score is LOW (<0.50) and best_config width was also used in a bad round.
- Use PER-DATASET configs — do not apply a single uniform config to all 4 datasets.

DIVERGENCE RULE (based on observable trace signal only):
- If a dataset trace shows final_loss > 3 × initial_loss → that dataset diverged.
- Next round: set lr=0.005 for that dataset (halve the default). Do not increase steps.
- The env_api will auto-retry internally on divergence, but lower lr in your config prevents it.

SAFE MATH RULE (prevents OOD-2 NaN collapse):
- safe_sqrt and safe_log are available: from env_api.kan_env import safe_sqrt, safe_log
- safe_sqrt(x) = sqrt(max(x, 0))  — prevents NaN when sqrt argument goes negative OOD
- safe_log(x)  = log(max(x, 1e-8)) — prevents NaN/−inf when log argument goes to 0 OOD
- USE safe_sqrt/safe_log in discover_law() whenever the expression contains:
    sqrt(sin(...)), sqrt(1 - ...), sqrt(exp(...) - constant), log(f(x))
  i.e. any sqrt/log whose argument can plausibly be negative outside training domain.
- In predict(), safe_sqrt/safe_log are also available via the same import.
- If feedback reports OOD2 < 0.20 AND expression contains sqrt or log: apply this rule.

Example of per-dataset loop:
configs = {
    'A': {'width': [2,2,1], 'steps': 400, 'lambda_reg': 0.001},
    'B': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.001},
    'C': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.001},
    'D': {'width': [2,3,1], 'steps': 400, 'lambda_reg': 0.001},
}
for did in ['A','B','C','D']:
    cfg = configs[did]
    model = KAN(width=cfg['width'], grid=3, k=3)
    # Phase 1: initial training
    trained_model, _ = train_kan(did, model, {'lr':0.01,'steps':cfg['steps'],'lambda_reg':cfg['lambda_reg']})
    # Phase 2: refine BEFORE auto_symbolic (increases spline resolution)
    trained_model = safe_refine(trained_model, 5)
    trained_model, _ = train_kan(did, trained_model, {'lr':0.001,'steps':200,'lambda_reg':cfg['lambda_reg']})
    # Phase 3: symbolise on precise splines
    try: trained_model.auto_symbolic(lib=['x','x^2','sin','exp','sqrt'])
    except Exception as e: print(f'auto_symbolic {did}: {e}')
    ...

RULES:
- KAN width MUST be [2,N,1] or [2,N,M,1]. width[0]=2, width[-1]=1. Any other -> crash.
- Never use: gplearn, PySR, eureqa, deap.

SCRIPT TEMPLATE (copy exactly every time):
import sys, os, numpy as np
sys.path.insert(0, '.')
os.environ.setdefault('DATA_DIR', 'data')
os.environ.setdefault('SOLUTION_DIR', 'solution')
os.environ.setdefault('TRACE_DIR', '/tmp/kan_traces')
from kan import KAN
from env_api.kan_env import train_kan, extract_symbolic, save_model, predict_from_expr, safe_refine, safe_sqrt, safe_log
os.makedirs('solution/models', exist_ok=True)
expressions = {}
for did in ['A','B','C','D']:
    model = KAN(width=[2,3,1], grid=3, k=3)
    # Phase 1: initial training on grid=3
    trained_model, _ = train_kan(did, model, {'lr':0.01,'steps':400,'lambda_reg':0.001})
    # Phase 2: refine FIRST (grid 3→5), then retrain — BEFORE auto_symbolic
    trained_model = safe_refine(trained_model, 5)
    trained_model, _ = train_kan(did, trained_model, {'lr':0.001,'steps':200,'lambda_reg':0.0001})
    # Phase 3: symbolise on precise splines — AFTER refine+retrain
    try: trained_model.auto_symbolic(lib=['x','x^2','sin','exp','sqrt'])
    except Exception as e: print(f'auto_symbolic {did}: {e}')
    save_model(trained_model, did)
    expressions[did] = extract_symbolic(trained_model, n_vars=2)
    print(f'{did}: {expressions[did]}')
_disc = f'''import sys, os, numpy as np
sys.path.insert(0, '.')
os.environ.setdefault('DATA_DIR', 'data')
os.environ.setdefault('SOLUTION_DIR', 'solution')
os.environ.setdefault('TRACE_DIR', '/tmp/kan_traces')
from env_api.kan_env import predict_from_expr
_EXPRESSIONS = {{
    "A": {repr(str(expressions.get("A", "0")))},
    "B": {repr(str(expressions.get("B", "0")))},
    "C": {repr(str(expressions.get("C", "0")))},
    "D": {repr(str(expressions.get("D", "0")))},
}}
def discover_law(dataset_id): return _EXPRESSIONS[dataset_id]
def predict(dataset_id, X): return predict_from_expr(_EXPRESSIONS[dataset_id], X)
'''
with open('solution/discover.py', 'w') as f: f.write(_disc)
print('discover.py written')

Output ONE python code block only. No explanation. Max 55 lines.\
"""


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(messages: list, retries: int = 3) -> str:
    # Keep only last 4 messages (2 exchanges) to limit input tokens
    trimmed = messages[-4:] if len(messages) > 4 else messages
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}",
               "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_API_URL, json=payload,
                                 headers=headers, timeout=120)
            if resp.status_code == 429:
                wait = 90 * (attempt + 1)
                print(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(15)
    raise RuntimeError("All retries exhausted")


# ── Code execution ────────────────────────────────────────────────────────────

def execute_code(code: str, timeout: int = 300) -> dict:
    with open("/tmp/agent_step.py", "w") as f:
        f.write(code)
    try:
        r = subprocess.run(
            [sys.executable, "/tmp/agent_step.py"],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONPATH": os.getcwd(),
                 "DATA_DIR": DATA_DIR, "SOLUTION_DIR": SOLUTION_DIR,
                 "TRACE_DIR": TRACE_DIR, "MPLBACKEND": "Agg"}
        )
        return {"stdout": r.stdout[-2000:], "stderr": r.stderr[-1000:],
                "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "```python" in text:
        candidate = text[text.rindex("```python") + 9:].strip()
        if len(candidate) > 20:
            return candidate
    return None


def filter_stdout(raw: str) -> str:
    """Strip verbose pykan lines to save context tokens."""
    skip = ("fixing (", "skipping (", "saving model")
    lines = [l for l in raw.splitlines()
             if not any(l.startswith(s) for s in skip)]
    return "\n".join(lines)[-1000:]


# ── Feedback ──────────────────────────────────────────────────────────────────

def load_rl_feedback() -> dict:
    if os.path.exists(FEEDBACK_PATH):
        try:
            return json.load(open(FEEDBACK_PATH))
        except Exception:
            pass
    return {}


def build_rl_context(feedback: dict) -> str:
    if not feedback:
        return ""
    prev  = feedback.get("final_score", 0)
    hist  = feedback.get("history_scores", [])
    prio  = feedback.get("priority_order", list("ABCD"))
    dsets = feedback.get("datasets", {})
    lines = [f"score={prev} history={hist}", f"fix_order={prio}", ""]
    for did in prio:
        d  = dsets.get(did, {})
        sc = d.get("score", 0)
        dt = d.get("delta", 0)
        m  = d.get("metrics", {})
        ac = d.get("advice", [])
        ood2 = m.get('ood2_score', m.get('ood2', 0))
        iid  = m.get('iid_score',  m.get('iid',  0))
        gap  = round(iid - ood2, 2)
        lines.append(
            f"  {did}: score={sc} ({dt:+.3f}) "
            f"iid={iid:.2f} ood2={ood2:.2f} gap={gap} "
            f"ast={m.get('ast_nodes','?')}"
        )
        for a in ac[:2]:
            lines.append(f"    -> {a}")
    return "\n".join(lines)


def build_initial_task() -> str:
    feedback = load_rl_feedback()
    ctx = build_rl_context(feedback)
    if ctx:
        return (f"Round {RL_ROUND}. Results:\n{ctx}\n\n"
                "Change ONE hyperparameter at a time. "
                "Simpler expressions score higher (exp(-0.03*complexity), cap=80 nodes). "
                "Output ONE python code block.")
    return ("Round 1. Use the SCRIPT TEMPLATE. "
            "Output ONE python code block.")


def feedback_prompt(iteration: int, exec_result: dict, traces: dict) -> str:
    rc  = exec_result.get("returncode", -1)
    out = filter_stdout(exec_result.get("stdout", ""))
    err = exec_result.get("stderr", "")[-500:] if rc != 0 else ""
    sol = os.path.exists(os.path.join(SOLUTION_DIR, "discover.py"))

    t_lines = []
    for did, t in traces.items():
        if t and t.get("loss_history"):
            lh = t["loss_history"]
            t_lines.append(f"  {did}: {lh[0]:.3f}->{lh[-1]:.3f} ok={t.get('kan_verified','?')}")

    parts = [f"iter={iteration}/{MAX_ITERATIONS} rc={rc} discover={'OK' if sol else 'MISSING'}"]
    if t_lines:
        parts.append("\n".join(t_lines))
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"errors:\n{err}")
    if not sol:
        parts.append("Write solution/discover.py using template.")
    parts.append(f"iters_left={MAX_ITERATIONS - iteration}. ONE python code block.")
    return "\n".join(parts)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent():
    feedback = load_rl_feedback()
    print("=" * 60)
    print(f"KAN Agent | {GROQ_MODEL} | round={RL_ROUND} | max_iter={MAX_ITERATIONS}")
    if feedback:
        print(f"prev={feedback.get('final_score','?')} history={feedback.get('history_scores',[])}")
    print("=" * 60)

    os.makedirs(SOLUTION_DIR, exist_ok=True)
    # Delete models dir every round to prevent stale model reuse
    import shutil
    models_dir = os.path.join(SOLUTION_DIR, "models")
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    os.makedirs(models_dir)

    messages     = []
    last_exec    = {}
    initial_task = build_initial_task()

    TRUNC_ENDS = (",", "(", "\\", "except", "Exceptio", "Exception",
                  "return", "def ", "for ", "if ", "try:", "    k")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─'*40} ITER {iteration}/{MAX_ITERATIONS} {'─'*40}")

        if iteration == 1:
            user_msg = initial_task
        else:
            traces = {}
            for did in "ABCD":
                tp = os.path.join(TRACE_DIR, f"trace_{did}.json")
                traces[did] = json.load(open(tp)) if os.path.exists(tp) else {}
            user_msg = feedback_prompt(iteration, last_exec, traces)

        messages.append({"role": "user", "content": user_msg})

        print(f"Calling {GROQ_MODEL}...")
        try:
            response = call_llm(messages)
        except Exception as e:
            print(f"API error: {e}")
            if iteration == 1:
                return False
            break

        print(f"Response: {len(response)} chars")
        messages.append({"role": "assistant", "content": response})

        code = extract_code(response)
        if not code:
            print("No code block found.")
            continue

        # Detect truncated code
        if any(code.rstrip().endswith(s) for s in TRUNC_ENDS):
            print(f"Truncated (ends={code.rstrip()[-40:]!r}). Asking for shorter.")
            messages.append({"role": "user", "content":
                "Code was TRUNCATED. Write SHORTER: max 50 lines, use compact loop. "
                "ONE python code block."})
            continue

        print(f"Exec ({len(code)} chars)...")
        print(f"--- code ---\n{code[:900]}\n--- end ---")
        last_exec = execute_code(code)
        rc = last_exec["returncode"]
        print(f"RC={rc}")
        if last_exec["stdout"]:
            print(f"stdout:\n{last_exec['stdout'][:700]}")
        if last_exec["stderr"] and rc != 0:
            print(f"stderr:\n{last_exec['stderr'][:400]}")

        time.sleep(2)

    solution_exists = os.path.exists(os.path.join(SOLUTION_DIR, "discover.py"))
    traces_exist    = any(os.path.exists(os.path.join(TRACE_DIR, f"trace_{d}.json"))
                          for d in "ABCD")
    solution_ok = solution_exists and traces_exist

    if solution_exists and not traces_exist:
        print("discover.py found but no KAN traces")

    json.dump({"model": GROQ_MODEL, "iterations": MAX_ITERATIONS,
               "solution_written": solution_ok, "traces_written": traces_exist},
              open(os.path.join(SOLUTION_DIR, "agent_log.json"), "w"), indent=2)

    print(f"\n{'='*60}\nAgent done. Solution: {'OK' if solution_ok else 'FAIL'}")
    return solution_ok


if __name__ == "__main__":
    sys.exit(0 if run_agent() else 1)