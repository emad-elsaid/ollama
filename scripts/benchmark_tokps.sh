#!/usr/bin/env bash
# End-to-end tokens/second benchmark for Ollama models

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODEL="${MODEL:-qwen2.5-coder:1.5b}"
PROMPT="${PROMPT:-Write a detailed explanation of the bubble sort algorithm with complete code examples in Python. Include time complexity analysis.}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} $*"; }

# Stop existing ollama
if pgrep -x ollama >/dev/null; then
	warn "Stopping existing Ollama..."
	pkill ollama || true
	sleep 2
fi

# Start server
log "Starting Ollama server..."
OLLAMA_NEW_ENGINE=1 ollama serve >/tmp/ollama_bench.log 2>&1 &
SERVER_PID=$!
sleep 3

cleanup() {
	log "Shutting down..."
	kill $SERVER_PID 2>/dev/null || true
	wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for server
for i in {1..30}; do
	if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
		log "Server ready"
		break
	fi
	[[ $i -eq 30 ]] && {
		echo "Server failed"
		exit 1
	}
	sleep 1
done

# Pull model
log "Ensuring model available..."
ollama pull "$MODEL" >/dev/null 2>&1 || true

# Warm-up
if [[ $WARMUP_RUNS -gt 0 ]]; then
	log "Warm-up ($WARMUP_RUNS run(s))..."
	for _ in $(seq 1 $WARMUP_RUNS); do
		echo "test" | ollama run "$MODEL" >/dev/null 2>&1
	done
fi

# Benchmark
log "Running $BENCHMARK_RUNS benchmark run(s)..."
RESULTS=()

for run in $(seq 1 $BENCHMARK_RUNS); do
	log "Run $run/$BENCHMARK_RUNS..."

	# Capture verbose output
	OUTPUT=$(echo "$PROMPT" | ollama run "$MODEL" --verbose 2>&1)

	# Parse metrics (example output: "eval count: 500")
	EVAL_COUNT=$(echo "$OUTPUT" | awk '/^eval count:/ {print $3}')
	EVAL_DURATION=$(echo "$OUTPUT" | awk '/^eval duration:/ {print $3}')

	# Convert duration to seconds
	if [[ "$EVAL_DURATION" == *"ms" ]]; then
		EVAL_SEC=$(echo "$EVAL_DURATION" | sed 's/ms//' | awk '{print $1 / 1000}')
	elif [[ "$EVAL_DURATION" == *"µs" ]]; then
		EVAL_SEC=$(echo "$EVAL_DURATION" | sed 's/µs//' | awk '{print $1 / 1000000}')
	else
		EVAL_SEC=$(echo "$EVAL_DURATION" | sed 's/s//')
	fi

	# Calculate tok/s
	if [[ -n "$EVAL_COUNT" ]] && [[ -n "$EVAL_SEC" ]] && (($(awk "BEGIN {print ($EVAL_SEC > 0)}"))); then
		TOK_S=$(awk "BEGIN {printf \"%.2f\", $EVAL_COUNT / $EVAL_SEC}")
	else
		TOK_S="0.00"
	fi

	RESULTS+=("$TOK_S")

	log "  Tokens: $EVAL_COUNT, Duration: $EVAL_DURATION, Throughput: $TOK_S tok/s"
done

# Summary
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Benchmark Summary"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Model: $MODEL"
for i in "${!RESULTS[@]}"; do
	log "Run $((i + 1)): ${RESULTS[$i]} tok/s"
done

# Average
AVG=$(printf '%s\n' "${RESULTS[@]}" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
log "Average: $AVG tok/s"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
