#!/usr/bin/env bash
# Orchestrates PLAN -> BUILD -> REVIEW -> FIX across Claude Code and Pi/DeepSeek.
#
# Usage: ./run-loop.sh "<task description>"
# Example: ./run-loop.sh "Implement roadmap item #3: POST /login with JWT issuance"
#
# The PLAN phase decides the task directory itself (see AGENTS.md PLAN role)
# and appends its result to status.md. Every phase after that reads the last
# line of status.md to find the current task dir and the previous phase's
# outcome, per the format:
#   TASK_DIR|PHASE|STATE|note
#
# This version prints every command it runs, its exit code, and how long it
# took, plus the current status.md tail after each phase — so you can tell
# at a glance whether it's actually progressing or stuck.
#
# Requires: claude (Claude Code CLI), pi (pi.dev), git

set -uo pipefail
# NOTE: not using -e. Failures are checked explicitly per-step so we can log
# a clear reason before exiting, instead of the script dying silently mid-phase.

TASK_DESC="${1:?Usage: $0 \"<task description>\"}"
MAX_FIX_ROUNDS=2
STATUS_FILE="status.md"

trap 'echo ""; log "Interrupted by user (Ctrl+C) - stopping."; exit 130' INT

CLAUDE_MODEL="claude-sonnet-5"
CLAUDE_ESCALATION_MODEL="claude-opus-5"
DEEPSEEK_MODEL="deepseek-v4-flash"

# --- logging helpers ---

ts() { date +"%H:%M:%S"; }

banner() {
  echo ""
  echo "==================================================================="
  echo "[$(ts)] $*"
  echo "==================================================================="
}

log() { echo "[$(ts)] $*"; }

show_status_tail() {
  if [ -f "$STATUS_FILE" ]; then
    log "status.md tail:"
    tail -n 3 "$STATUS_FILE" | sed 's/^/    /'
  else
    log "status.md does not exist yet."
  fi
}

# Runs a command, printing what it is, streaming its output live, and
# reporting exit code + duration. Exits the script on failure.
run_cmd() {
  local desc="$1"
  shift
  log "-> ${desc}"
  log "   \$ $*"
  local start end dur rc
  start=$(date +%s)
  "$@"
  rc=$?
  end=$(date +%s)
  dur=$((end - start))
  if [ "$rc" -eq 0 ]; then
    log "OK  ${desc} (${dur}s)"
  else
    log "FAIL  ${desc}  exit=${rc}  (${dur}s)"
    log "Stopping here so you can inspect the repo / status.md before retrying."
    exit "$rc"
  fi
}

commit() {
  local msg="$1"
  git add -A
  if git commit -m "$msg" --quiet; then
    log "committed: ${msg}"
  else
    log "nothing to commit for: ${msg}"
  fi
}

# Reads the last line of status.md and splits it into TASK_DIR / PHASE / STATE / NOTE.
last_status_line() {
  if [ ! -f "$STATUS_FILE" ]; then
    log "ERROR: ${STATUS_FILE} not found."
    exit 1
  fi
  local line
  line="$(tail -n1 "$STATUS_FILE")"
  if [ -z "$line" ]; then
    log "ERROR: ${STATUS_FILE} is empty."
    exit 1
  fi
  IFS='|' read -r TASK_DIR PHASE STATE NOTE <<< "$line"
  if [ -z "${TASK_DIR:-}" ] || [ -z "${PHASE:-}" ] || [ -z "${STATE:-}" ]; then
    log "ERROR: could not parse last status.md line: '${line}'"
    exit 1
  fi
  log "parsed status.md -> TASK_DIR=${TASK_DIR} PHASE=${PHASE} STATE=${STATE} NOTE=${NOTE:-}"
}

# Counts consecutive trailing FIX|BLOCKED entries for the current TASK_DIR,
# used to decide when to escalate to Claude Opus.
consecutive_blocked_fixes() {
  local dir="$1"
  tac "$STATUS_FILE" | awk -F'|' -v dir="$dir" '
    $1 == dir && $2 == "FIX" && $3 == "BLOCKED" { c++; next }
    { exit }
    END { print c+0 }
  '
}

# --- phases ---

plan_phase() {
  banner "PLAN  (${CLAUDE_MODEL})"
  run_cmd "claude PLAN" claude --dangerously-skip-permissions -p "Read AGENTS.md for the PLAN role. Task: ${TASK_DESC}. Follow the status.md logging contract exactly (append TASK_DIR|PLAN|CREATED|<summary> as the final line)."
  last_status_line
  if [ "$PHASE" != "PLAN" ] || [ "$STATE" != "CREATED" ]; then
    log "ERROR: PLAN did not append the expected status.md line."
    exit 1
  fi
  log "task directory resolved: ${TASK_DIR}"
  commit "${TASK_DIR}: plan"
  show_status_tail
}

build_phase() {
  banner "BUILD  (${DEEPSEEK_MODEL} via pi)"
  run_cmd "pi BUILD" pi --provider deepseek --model "$DEEPSEEK_MODEL" -p \
    "Read AGENTS.md for the BUILD role. Read the last line of status.md to confirm the current task directory (${TASK_DIR}). Read ${TASK_DIR}/plan.md. Implement it. Append TASK_DIR|BUILD|DONE|<summary> to status.md when finished."
  commit "${TASK_DIR}: build"
  show_status_tail
}

review_phase() {
  banner "REVIEW  (${CLAUDE_MODEL})"
  run_cmd "claude REVIEW" claude --dangerously-skip-permissions -p "Read AGENTS.md for the REVIEW role. The current task directory is ${TASK_DIR}. Diff recent commits against ${TASK_DIR}/plan.md, run the test suite yourself, and write findings to ${TASK_DIR}/review.md. Append ${TASK_DIR}|REVIEW|CLEAN or ${TASK_DIR}|REVIEW|ISSUES|<n findings> to status.md."
  last_status_line
  commit "${TASK_DIR}: review"
  show_status_tail
}

fix_phase() {
  banner "FIX  round ${round}  (${DEEPSEEK_MODEL} via pi)"
  run_cmd "pi FIX" pi --provider deepseek --model "$DEEPSEEK_MODEL" -p \
    "Read AGENTS.md for the FIX role. Confirm via the last line of status.md that the current task (${TASK_DIR}) is in REVIEW|ISSUES state before proceeding. Read ${TASK_DIR}/review.md and fix each item, re-running tests after each fix. Append ${TASK_DIR}|FIX|FIXED|<summary> or ${TASK_DIR}|FIX|BLOCKED|<what's still failing> to status.md."
  commit "${TASK_DIR}: fix (round ${round})"
  show_status_tail
}

escalate_phase() {
  banner "ESCALATE  (${CLAUDE_ESCALATION_MODEL})"
  run_cmd "claude ESCALATE" claude --dangerously-skip-permissions --model "$CLAUDE_ESCALATION_MODEL" -p \
    "Read AGENTS.md. Task ${TASK_DIR} has ${MAX_FIX_ROUNDS} consecutive FIX|BLOCKED entries in status.md. Read ${TASK_DIR}/review.md and the relevant status.md history, then resolve the issue directly. Append ${TASK_DIR}|FIX|FIXED|<summary> (escalated) to status.md."
  commit "${TASK_DIR}: escalated fix"
  show_status_tail
}

# --- main loop ---

log "starting run-loop.sh"
log "task description: ${TASK_DESC}"

plan_phase
build_phase
review_phase

round=0
while [ "$PHASE" = "REVIEW" ] && [ "$STATE" = "ISSUES" ]; do
  round=$((round + 1))
  blocked_streak=$(consecutive_blocked_fixes "$TASK_DIR")
  log "review found issues. consecutive blocked fixes so far: ${blocked_streak}/${MAX_FIX_ROUNDS}"

  if [ "$blocked_streak" -ge "$MAX_FIX_ROUNDS" ]; then
    log "fix rounds exhausted -> escalating to Claude Opus"
    escalate_phase
    review_phase
    if [ "$PHASE" = "REVIEW" ] && [ "$STATE" = "ISSUES" ]; then
      banner "STILL UNRESOLVED AFTER ESCALATION -- stopping for human review"
      exit 1
    fi
    break
  fi

  fix_phase
  review_phase
done

if [ "$PHASE" = "REVIEW" ] && [ "$STATE" = "CLEAN" ]; then
  banner "DONE -- task ${TASK_DIR} complete, review is CLEAN"
else
  banner "WARNING -- loop ended in unexpected state: ${TASK_DIR}|${PHASE}|${STATE}"
  exit 1
fi