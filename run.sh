#!/bin/bash
# run.sh — KAN RL Environment pipeline
#
# Modes:
#   full          — RL loop: datagen → N rounds of (agent → judge → feedback)
#   agent-only    — single agent run (uses rl_feedback.json if present)
#   judge-only    — run judge on existing solution
#   data-only     — generate datasets only
#
# RL loop env vars (set in .env):
#   RL_ROUNDS      — number of outer RL rounds (default: 3)
#   RL_ROUND       — current round number (set automatically by this script)
#   MAX_ITERATIONS — agent inner iterations per round (default: 4)

set -e
MODE=${1:-"full"}

echo "============================================================"
echo "  KAN Symbolic Regression RL Environment"
echo "  Mode: $MODE"
echo "============================================================"

# ── Generate datasets (once per run) ─────────────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "data-only" ]]; then
    echo ""
    echo "[1/3] Generating datasets..."
    python3 datasets/generate.py
    echo "Done."
fi

if [[ "$MODE" == "data-only" ]]; then
    echo ""; echo "============================================================"
    echo "  Data generation complete."; echo "============================================================"
    exit 0
fi

# ── Judge only ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "judge-only" ]]; then
    echo ""; echo "[judge] Running judge..."
    python3 judge/judge.py
    echo ""; echo "Judge result saved to solution/judge_result.json"
    echo ""; echo "============================================================"
    echo "  Pipeline complete."; echo "============================================================"
    exit 0
fi

# ── Agent only ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "agent-only" ]]; then
    [[ -z "$GROQ_API_KEY" ]] && echo "⚠️  GROQ_API_KEY not set." && exit 1
    echo ""; echo "[agent] Running agent (round ${RL_ROUND:-1})..."
    export RL_ROUND=${RL_ROUND:-1}
    python3 agent/agent.py
    exit $?
fi

# ── Full RL loop ──────────────────────────────────────────────────────────────
if [[ "$MODE" == "full" ]]; then

    if [[ -z "$GROQ_API_KEY" ]]; then
        echo ""; echo "⚠️  GROQ_API_KEY not set — running judge only on existing solution."
        python3 judge/judge.py
        exit 0
    fi

    # ── Clean previous run state ──────────────────────────────────────────────
    # Critical: stale feedback/traces from a previous run corrupt the RL loop.
    # - rl_feedback.json / rl_history.json : agent would think it's round N+1
    # - judge_result.json                  : stale score would set wrong BEST_SCORE
    # - /tmp/kan_traces/*.json             : judge uses these to verify train_kan()
    #                                        was called THIS round — stale = false pass
    # We keep solution/models/ so the agent can optionally reuse trained weights.
    echo ""; echo "  [init] Cleaning previous run state..."
    rm -f solution/rl_feedback.json \
          solution/rl_history.json \
          solution/judge_result.json \
          solution/discover_best.py \
          solution/agent_log.json
    rm -f /tmp/kan_traces/trace_*.json
    echo "  [init] Clean. Starting fresh RL loop."

    RL_ROUNDS=${RL_ROUNDS:-3}
    PASS_THRESHOLD=0.65
    BEST_SCORE=0

    echo ""
    echo "[2/3] RL Loop — ${RL_ROUNDS} rounds × ${MAX_ITERATIONS:-4} agent iterations"
    echo "────────────────────────────────────────────────────────────"

    for ROUND in $(seq 1 $RL_ROUNDS); do

        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        printf  "║  RL ROUND %-3s / %-3s                                     ║\n" "$ROUND" "$RL_ROUNDS"
        echo "╚══════════════════════════════════════════════════════════╝"

        # Clear KAN traces from previous round — judge uses these to verify
        # train_kan() was called THIS round. Stale traces = false kan_conf score.
        # Also remove discover.py — a stale/broken file from a crashed round
        # would be scored by the judge instead of the new solution.
        rm -f /tmp/kan_traces/trace_*.json
        rm -f solution/discover.py

        # Run agent
        echo ""; echo "  [agent] Running LLM agent..."
        export RL_ROUND=$ROUND
        python3 agent/agent.py || true   # don't abort if agent fails

        # Run judge (always — even if agent failed, score existing solution)
        echo ""; echo "  [judge] Scoring solution..."
        python3 judge/judge.py || true

        # Read score
        SCORE=$(python3 -c "
import json
try:
    r = json.load(open('solution/judge_result.json'))
    print(f\"{r.get('final_score', 0):.4f}\")
except:
    print('0.0000')
")
        echo ""
        echo "  Round ${ROUND} score: ${SCORE}  (threshold: ${PASS_THRESHOLD})"

        # Keep best solution
        IS_BETTER=$(python3 -c "print('yes' if float('${SCORE}') > float('${BEST_SCORE}') else 'no')")
        if [[ "$IS_BETTER" == "yes" ]]; then
            BEST_SCORE=$SCORE
            cp solution/discover.py solution/discover_best.py 2>/dev/null || true
            echo "  ✨ New best: ${BEST_SCORE} — saved to solution/discover_best.py"
        fi

        # Check pass
        PASSED=$(python3 -c "print('yes' if float('${SCORE}') >= ${PASS_THRESHOLD} else 'no')")
        if [[ "$PASSED" == "yes" ]]; then
            echo ""; echo "  ✅ PASSED threshold ${PASS_THRESHOLD} at round ${ROUND}!"
            break
        fi

        # Format feedback for next round + wait for Groq rate limit reset
        if [[ $ROUND -lt $RL_ROUNDS ]]; then
            echo ""; echo "  [feedback] Formatting RL feedback for round $((ROUND+1))..."
            python3 agent/feedback_formatter.py $ROUND
            INTER_ROUND_WAIT=${INTER_ROUND_WAIT:-120}
            echo "  [rate-limit] Waiting ${INTER_ROUND_WAIT}s before next round (Groq token quota reset)..."
            sleep $INTER_ROUND_WAIT
        fi

    done

    # Restore best solution if last round was worse
    if [[ -f "solution/discover_best.py" ]]; then
        CURRENT=$(python3 -c "
import json
try:
    r = json.load(open('solution/judge_result.json'))
    print(f\"{r.get('final_score', 0):.4f}\")
except:
    print('0.0000')
")
        RESTORE=$(python3 -c "print('yes' if float('${BEST_SCORE}') > float('${CURRENT}') else 'no')")
        if [[ "$RESTORE" == "yes" ]]; then
            cp solution/discover_best.py solution/discover.py
            echo "  ♻️  Restored best solution (${BEST_SCORE} > ${CURRENT})"
            python3 judge/judge.py  # re-score best solution
        fi
    fi

    echo ""
    echo "[3/3] RL loop complete. Best score: ${BEST_SCORE}"

fi

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"