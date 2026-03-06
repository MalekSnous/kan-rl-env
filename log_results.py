"""
log_results.py
--------------
Parses RL loop docker logs and saves structured results per model.

Usage:
    # Pendant ou après une run, pipe les logs vers ce script :
    docker compose logs kan-rl-env | python log_results.py --model "gpt-oss-120b"

    # Ou depuis un fichier log sauvegardé :
    python log_results.py --model "llama-3.3-70b" --logfile logs/llama70b_run1.txt

    # Combiner plusieurs runs pour le plot :
    python log_results.py --plot  # lit tous les JSON dans results/

Output:
    results/{model_name}.json   — scores structurés par round
    results/all_results.csv     — CSV consolidé pour tous les modèles
    plots/                      — graphiques PNG générés par --plot
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

MODELS_DISPLAY = {
    "gpt-oss-120b":        "GPT-OSS 120B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant":    "Llama 3.1 8B",
}

DATASET_COLORS = {
    "A": "#2196F3",
    "B": "#4CAF50",
    "C": "#FF9800",
    "D": "#E91E63",
}

MODEL_COLORS = {
    "GPT-OSS 120B":  "#1565C0",
    "Llama 3.3 70B": "#2E7D32",
    "Llama 3.1 8B":  "#E65100",
}

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_logs(text: str) -> list[dict]:
    """
    Parse docker log text and return a list of round records.
    Each record: {round, model, final_score, passed, datasets: {A,B,C,D: {...}}}
    """
    rounds = []
    current_round = None
    current_model = None

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect round header
        m = re.search(r"RL ROUND\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            current_round = int(m.group(1))
            i += 1
            continue

        # Detect model (also extract round number as fallback)
        m = re.search(r"KAN Agent \| ([^\|]+) \| round=(\d+)", line)
        if m:
            raw_model = m.group(1).strip()
            current_model = MODELS_DISPLAY.get(raw_model, raw_model)
            if current_round is None:
                current_round = int(m.group(2))
            i += 1
            continue

        # Detect FINAL SCORE
        m = re.search(r"FINAL SCORE:\s*([\d.]+)\s*\((PASS|FAIL)", line)
        if m and current_round is not None:
            final_score = float(m.group(1))
            passed      = m.group(2) == "PASS"

            # Collect per-dataset scores from next ~20 lines
            datasets = {}
            j = i + 1
            while j < min(i + 30, len(lines)):
                # Dataset score line: "  Dataset X: 0.7777"
                dm = re.search(r"Dataset\s+([ABCD]):\s*([\d.]+)", lines[j])
                if dm:
                    datasets[dm.group(1)] = {"score": float(dm.group(2))}
                j += 1

            # Back-fill gen/parsimony/consistency/kan_conf/ast_nodes from earlier lines
            # Search backwards for Score[X] lines
            for k in range(max(0, i - 80), i):
                sm = re.search(
                    r"Score\[([ABCD])\]\] gen=([\d.]+) \| parsimony=([\d.]+) \| "
                    r"consistency=([\d.]+) \| kan_conf=([\d.]+)",
                    lines[k]
                )
                if sm:
                    did = sm.group(1)
                    if did in datasets:
                        datasets[did].update({
                            "gen":         float(sm.group(2)),
                            "parsimony":   float(sm.group(3)),
                            "consistency": float(sm.group(4)),
                            "kan_conf":    float(sm.group(5)),
                        })
                # AST nodes
                am = re.search(r"Step3\[([ABCD])\]\] AST nodes=(\d+)", lines[k])
                if am:
                    did = am.group(1)
                    if did in datasets:
                        datasets[did]["ast_nodes"] = int(am.group(2))

                # OOD scores
                om = re.search(
                    r"Step5\[([ABCD])\]\[(iid|ood1|ood2)\]\] .*score=([\d.]+)", lines[k]
                )
                if om:
                    did    = om.group(1)
                    regime = om.group(2)
                    score  = float(om.group(3))
                    if did in datasets:
                        datasets[did][f"{regime}_score"] = score

            rounds.append({
                "round":       current_round,
                "model":       current_model,
                "final_score": final_score,
                "passed":      passed,
                "datasets":    datasets,
            })
            i = j
            continue

        i += 1

    return rounds


# ── Save / load ───────────────────────────────────────────────────────────────

def save_results(model_name: str, rounds: list[dict]):
    safe_name = re.sub(r"[^\w\-]", "_", model_name)
    path = RESULTS_DIR / f"{safe_name}.json"
    data = {
        "model":      model_name,
        "saved_at":   datetime.now().isoformat(),
        "n_rounds":   len(rounds),
        "best_score": max((r["final_score"] for r in rounds), default=0),
        "passed":     any(r["passed"] for r in rounds),
        "rounds":     rounds,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {path}  ({len(rounds)} rounds)")
    return path


def load_all_results() -> dict[str, dict]:
    results = {}
    for p in RESULTS_DIR.glob("*.json"):
        with open(p) as f:
            data = json.load(f)
        model = data["model"]
        results[model] = data
    return results


def export_csv(all_results: dict):
    """Export consolidated CSV for all models."""
    import csv
    path = RESULTS_DIR / "all_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "round", "final_score", "passed",
            "A_score", "B_score", "C_score", "D_score",
            "A_gen", "B_gen", "C_gen", "D_gen",
            "A_parsimony", "B_parsimony", "C_parsimony", "D_parsimony",
            "A_ast", "B_ast", "C_ast", "D_ast",
            "A_ood2", "B_ood2", "C_ood2", "D_ood2",
        ])
        for model, data in all_results.items():
            for r in data["rounds"]:
                ds = r["datasets"]
                row = [model, r["round"], r["final_score"], r["passed"]]
                for did in "ABCD":
                    row.append(ds.get(did, {}).get("score", ""))
                for did in "ABCD":
                    row.append(ds.get(did, {}).get("gen", ""))
                for did in "ABCD":
                    row.append(ds.get(did, {}).get("parsimony", ""))
                for did in "ABCD":
                    row.append(ds.get(did, {}).get("ast_nodes", ""))
                for did in "ABCD":
                    row.append(ds.get(did, {}).get("ood2_score", ""))
                writer.writerow(row)
    print(f"CSV exported → {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(all_results: dict):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "figure.dpi":       150,
    })

    THRESHOLD = 0.65

    # ── Plot 1: Final score evolution per model ───────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, data in all_results.items():
        rounds_x = [r["round"] for r in data["rounds"]]
        scores_y = [r["final_score"] for r in data["rounds"]]
        color = MODEL_COLORS.get(model, "#555")
        ax.plot(rounds_x, scores_y, "o-", color=color, linewidth=2,
                markersize=5, label=model, alpha=0.9)
        # Mark best
        best_idx = scores_y.index(max(scores_y))
        ax.annotate(f'{scores_y[best_idx]:.4f}',
                    (rounds_x[best_idx], scores_y[best_idx]),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=9, color=color, fontweight="bold")

    ax.axhline(THRESHOLD, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Threshold ({THRESHOLD})", alpha=0.8)
    ax.set_xlabel("RL Round")
    ax.set_ylabel("Final Score")
    ax.set_title("RL Score Evolution by Model", fontsize=14, fontweight="bold", pad=12)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    p = PLOTS_DIR / "01_score_evolution.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Plot → {p}")

    # ── Plot 2: Per-dataset score evolution (one subplot per model) ───────────
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, all_results.items()):
        rounds_x = [r["round"] for r in data["rounds"]]
        for did in "ABCD":
            scores = [r["datasets"].get(did, {}).get("score", None)
                      for r in data["rounds"]]
            valid = [(x, y) for x, y in zip(rounds_x, scores) if y is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "o-", color=DATASET_COLORS[did],
                        linewidth=1.8, markersize=4, label=f"Dataset {did}", alpha=0.85)

        ax.axhline(THRESHOLD, color="#E53935", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("RL Round")
        if ax == axes[0]:
            ax.set_ylabel("Dataset Score")
        ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Per-Dataset Score Evolution by Model",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = PLOTS_DIR / "02_per_dataset_evolution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {p}")

    # ── Plot 3: Best scores comparison table (bar chart) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: best final score per model
    ax = axes[0]
    models  = list(all_results.keys())
    bests   = [all_results[m]["best_score"] for m in models]
    colors  = [MODEL_COLORS.get(m, "#555") for m in models]
    bars = ax.bar(models, bests, color=colors, alpha=0.85, width=0.5)
    ax.axhline(THRESHOLD, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Threshold ({THRESHOLD})")
    for bar, val in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Best Score Achieved", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()

    # Right: per-dataset best score heatmap-style bar
    ax = axes[1]
    x     = list(range(len(models)))
    width = 0.2
    for idx, did in enumerate("ABCD"):
        best_per_model = []
        for m in models:
            scores = [r["datasets"].get(did, {}).get("score", 0)
                      for r in all_results[m]["rounds"]]
            best_per_model.append(max(scores) if scores else 0)
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, best_per_model, width,
                      label=f"Dataset {did}",
                      color=DATASET_COLORS[did], alpha=0.85)

    ax.axhline(THRESHOLD, color="#E53935", linestyle="--",
               linewidth=1.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Best Per-Dataset Score by Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    p = PLOTS_DIR / "03_best_scores_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Plot → {p}")

    # ── Plot 4: OOD-2 scores evolution (extrapolation quality) ───────────────
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, all_results.items()):
        rounds_x = [r["round"] for r in data["rounds"]]
        for did in "ABCD":
            ood2 = [r["datasets"].get(did, {}).get("ood2_score", None)
                    for r in data["rounds"]]
            valid = [(x, y) for x, y in zip(rounds_x, ood2) if y is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "s--", color=DATASET_COLORS[did],
                        linewidth=1.5, markersize=4,
                        label=f"Dataset {did}", alpha=0.85)

        ax.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("RL Round")
        if ax == axes[0]:
            ax.set_ylabel("OOD-2 Score (extrapolation)")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("OOD-2 Extrapolation Score by Model (hardest regime)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = PLOTS_DIR / "04_ood2_evolution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {p}")

    # ── Plot 5: Parsimony evolution ───────────────────────────────────────────
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, all_results.items()):
        rounds_x = [r["round"] for r in data["rounds"]]
        for did in "ABCD":
            pars = [r["datasets"].get(did, {}).get("parsimony", None)
                    for r in data["rounds"]]
            valid = [(x, y) for x, y in zip(rounds_x, pars) if y is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "^-", color=DATASET_COLORS[did],
                        linewidth=1.5, markersize=4,
                        label=f"Dataset {did}", alpha=0.85)

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("RL Round")
        if ax == axes[0]:
            ax.set_ylabel("Parsimony Score")
        ax.set_ylim(-0.05, 1.0)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Parsimony Score Evolution (expression simplicity)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = PLOTS_DIR / "05_parsimony_evolution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {p}")

    print(f"\n✅  {5} plots saved in {PLOTS_DIR}/")


# ── Summary table (terminal) ──────────────────────────────────────────────────

def print_summary(all_results: dict):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for model, data in all_results.items():
        rounds  = data["rounds"]
        best    = data["best_score"]
        passed  = data["passed"]
        n_pass  = sum(1 for r in rounds if r["passed"])
        best_r  = next((r["round"] for r in rounds
                        if r["final_score"] == best), "?")
        status  = "PASS ✅" if passed else "FAIL ❌"
        print(f"\n  {model:<22} {status}")
        print(f"  Best score: {best:.4f}  (round {best_r}/{len(rounds)})  "
              f"— {n_pass}/{len(rounds)} rounds passed")

        # Per-dataset best
        print(f"  {'Dataset':<10} {'Best':>6}  {'Best Gen':>9}  "
              f"{'Best OOD2':>10}  {'Best Pars':>10}")
        print(f"  {'-'*50}")
        for did in "ABCD":
            scores = [r["datasets"].get(did, {}).get("score", 0)    for r in rounds]
            gens   = [r["datasets"].get(did, {}).get("gen", 0)       for r in rounds]
            ood2s  = [r["datasets"].get(did, {}).get("ood2_score", 0) for r in rounds]
            pars   = [r["datasets"].get(did, {}).get("parsimony", 0) for r in rounds]
            print(f"  {did:<10} {max(scores):>6.4f}  {max(gens):>9.4f}  "
                  f"{max(ood2s):>10.4f}  {max(pars):>10.4f}")

    print("\n" + "=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KAN RL results logger & plotter")
    parser.add_argument("--model",   type=str, help="Model name for this run")
    parser.add_argument("--logfile", type=str, help="Path to log file (default: stdin)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate plots from all saved results")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table from all saved results")
    args = parser.parse_args()

    if args.plot or args.summary:
        all_results = load_all_results()
        if not all_results:
            print("No results found in results/. Run with --model first.")
            sys.exit(1)
        export_csv(all_results)
        if args.summary:
            print_summary(all_results)
        if args.plot:
            plot_all(all_results)
        return

    if not args.model:
        parser.error("--model is required when parsing logs")

    if args.logfile:
        with open(args.logfile) as f:
            text = f.read()
    else:
        print(f"Reading logs from stdin for model '{args.model}'...")
        text = sys.stdin.read()

    rounds = parse_logs(text)
    if not rounds:
        print("⚠️  No rounds found in logs. Check log format.")
        sys.exit(1)

    print(f"Parsed {len(rounds)} rounds for {args.model}")
    save_results(args.model, rounds)

    # Quick terminal summary
    best = max(r["final_score"] for r in rounds)
    passed = any(r["passed"] for r in rounds)
    print(f"Best score: {best:.4f}  |  Passed: {passed}")


if __name__ == "__main__":
    main()