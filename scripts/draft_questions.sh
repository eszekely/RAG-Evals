#!/usr/bin/env bash
# Draft evaluation questions: one fresh, headless Claude Code session per page range.
#
#   bash scripts/draft_questions.sh --dry-run     # print the filled-in prompts; costs nothing
#   bash scripts/draft_questions.sh --only 3      # run only range number 3 (test one first!)
#   bash scripts/draft_questions.sh               # run every range not yet drafted
#
# Environment: MODEL (default "fable"), MAX_BUDGET_USD per run (default 3).
#
# Why one session per range: each run sees only its own 20 pages (no memory of earlier
# questions), and its token usage is exactly one run's.
# Each run writes eval/drafts/<stem>_pp<A>-<B>_<model>_<date>.jsonl (the questions, exactly
# as generated) and a .usage.json next to it (tokens and cost of that run alone), so a
# re-run of one range never mixes with the others. A range that already has a draft is
# skipped; to redo one, delete its two files yourself.
# This script spends money (it calls the Claude API through Claude Code): run it yourself.

set -euo pipefail
cd "$(dirname "$0")/.."  # project root, so relative paths work from anywhere

MODEL="${MODEL:-fable}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-3}"
DATE="$(date +%F)"
TEMPLATE="scripts/draft_prompt.txt"
PAGES="data/processed/pages.jsonl"
OUT_DIR="eval/drafts"

# document, first page, last page (PDF page index, as in pages.jsonl). The last range of
# each report is merged with its short tail, since 10 questions need more than a few pages.
RANGES=(
  "sogwr_2025_WMO-1402_en.pdf 1 20"
  "sogwr_2025_WMO-1402_en.pdf 21 40"
  "sogwr_2025_WMO-1402_en.pdf 41 60"
  "sogwr_2025_WMO-1402_en.pdf 61 80"
  "sogwr_2025_WMO-1402_en.pdf 81 100"
  "sogwr_2025_WMO-1402_en.pdf 101 118"
  "sogwr_2024_WMO-1380_en.pdf 1 20"
  "sogwr_2024_WMO-1380_en.pdf 21 40"
  "sogwr_2024_WMO-1380_en.pdf 41 60"
  "sogwr_2024_WMO-1380_en.pdf 61 80"
  "sogwr_2024_WMO-1380_en.pdf 81 98"
  "sogwr_2023_WMO-1362_en.pdf 1 20"
  "sogwr_2023_WMO-1362_en.pdf 21 40"
  "sogwr_2023_WMO-1362_en.pdf 41 60"
  "sogwr_2023_WMO-1362_en.pdf 61 81"
  "sogwr_2022_WMO-1333_en.pdf 1 20"
  "sogwr_2022_WMO-1333_en.pdf 21 40"
  "sogwr_2022_WMO-1333_en.pdf 41 67"
)

dry_run=0
only=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --only) only="${2:?--only needs a range number from 1 to ${#RANGES[@]}}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

fill_prompt() {  # $1 file name, $2 first page, $3 last page -> prompt text on stdout
  local file="$1" a="$2" b="$3" stem year prompt
  stem="${file%.pdf}"
  year="${stem:6:4}"  # "sogwr_2024_..." -> 2024
  prompt="$(cat "$TEMPLATE")"
  prompt="${prompt//\{FILENAME\}/$file}"
  prompt="${prompt//\{STEM\}/$stem}"
  prompt="${prompt//\{YEAR\}/$year}"
  prompt="${prompt//\{A\}/$a}"
  prompt="${prompt//\{B\}/$b}"
  prompt="${prompt//\{MODEL\}/$MODEL}"
  prompt="${prompt//\{DATE\}/$DATE}"
  printf '%s\n' "$prompt"
}

valid_draft() {  # $1 path: 1-10 lines, each with the required fields and values
  jq -e -s '
    length >= 1 and length <= 10 and all(.[];
      (.id | type == "string") and (.question | type == "string")
      and (.type | IN("single_fact", "number", "table", "multi_hop"))
      and (.gold_pages | type == "array" and length > 0)
      and (.evidence_snippet | type == "string" and (split(" ") | length) <= 25)
      and .author == "llm" and .verified == false and .split == null)' "$1" >/dev/null
}

[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }
if [ "$dry_run" -eq 0 ]; then
  command -v claude >/dev/null || { echo "claude command not found" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }
  [ -f "$PAGES" ] || { echo "missing $PAGES: run python -m rag.ingest first" >&2; exit 1; }
  mkdir -p "$OUT_DIR"
fi

marker="$(mktemp)"
trap 'rm -f "$marker"' EXIT
failures=0
n=0
for entry in "${RANGES[@]}"; do
  n=$((n + 1))
  if [ -n "$only" ] && [ "$n" != "$only" ]; then continue; fi
  read -r file a b <<<"$entry"
  stem="${file%.pdf}"
  out="$OUT_DIR/${stem}_pp${a}-${b}_${MODEL}_${DATE}.jsonl"
  label="[$n/${#RANGES[@]}] $file pages $a-$b"

  if [ "$dry_run" -eq 1 ]; then
    printf '===== %s -> %s\n' "$label" "$out"
    fill_prompt "$file" "$a" "$b"
    continue
  fi
  if compgen -G "$OUT_DIR/${stem}_pp${a}-${b}_${MODEL}_*.jsonl" >/dev/null; then
    echo "$label: already drafted, skipped"
    continue
  fi

  echo "$label: running ($MODEL, budget \$$MAX_BUDGET_USD) ..."
  touch "$marker"
  # The prompt goes in on stdin. Allowed tools: read files, write ONLY under eval/drafts/,
  # and jq to filter pages.jsonl. acceptEdits lets the run create its file without asking.
  if ! response="$(fill_prompt "$file" "$a" "$b" | claude -p \
        --model "$MODEL" --output-format json --permission-mode acceptEdits \
        --allowedTools "Read" "Write(eval/drafts/**)" "Bash(jq:*)" \
        --max-budget-usd "$MAX_BUDGET_USD" --no-session-persistence)"; then
    echo "$label: FAILED (claude exited with an error)"
    failures=$((failures + 1))
    continue
  fi
  usage="${out%.jsonl}.usage.json"
  printf '%s' "$response" | jq 'del(.result)' >"$usage" || echo "  warning: no usage JSON saved"

  if [ ! -f "$out" ]; then
    echo "$label: FAILED (expected draft file was not written)"
    failures=$((failures + 1))
  elif ! valid_draft "$out"; then
    echo "$label: FAILED (draft is not valid: need 1-10 JSON lines with the required fields)"
    failures=$((failures + 1))
  else
    echo "$label: ok, $(wc -l <"$out" | tr -d ' ') questions"
  fi
  # The prompt allows creating one file only; flag anything else that changed.
  stray="$(find . -type f -newer "$marker" -not -path './.git/*' -not -path './.venv/*' \
            -not -path './eval/drafts/*' -not -path './.*_cache/*' -not -name '*.pyc' | head -5)"
  [ -z "$stray" ] || echo "  WARNING: files changed outside eval/drafts/:" $'\n'"$stray"
done

if [ "$dry_run" -eq 0 ] && compgen -G "$OUT_DIR/*_${MODEL}_${DATE}.usage.json" >/dev/null; then
  echo "Usage today for $MODEL (sum of the .usage.json files):"
  jq -s '{runs: length,
          input_tokens: (map(.usage.input_tokens // 0) | add),
          cache_write_tokens: (map(.usage.cache_creation_input_tokens // 0) | add),
          cache_read_tokens: (map(.usage.cache_read_input_tokens // 0) | add),
          output_tokens: (map(.usage.output_tokens // 0) | add),
          cost_usd: (map(.total_cost_usd // 0) | add)}' "$OUT_DIR"/*_"${MODEL}_${DATE}".usage.json
fi
[ "$failures" -eq 0 ] || { echo "$failures range(s) failed"; exit 1; }
