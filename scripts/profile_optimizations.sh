#!/usr/bin/env bash
# Automated profiling harness for Ollama native engine optimizations
# Captures CPU and memory profiles during model inference

set -euo pipefail

# Configuration
MODEL="${MODEL:-qwen2.5-coder:1.5b}"
PROMPT="${PROMPT:-Write a detailed explanation of the bubble sort algorithm}"
PROFILE_DIR="${PROFILE_DIR:-ollama-profiles}"
ENGINE="${OLLAMA_NEW_ENGINE:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
	echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

warn() {
	echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} $*"
}

error() {
	echo -e "${RED}[$(date +'%H:%M:%S')]${NC} $*" >&2
}

# Ensure profile directory exists
PROFILE_DIR="$(pwd)/$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

# Find next profile number
PROFILE_NUM=1
while [[ -f "$PROFILE_DIR/cpu_$(printf '%04d' $PROFILE_NUM).prof" ]]; do
	((PROFILE_NUM++))
done
PROFILE_ID=$(printf '%04d' $PROFILE_NUM)

CPU_PROF="$PROFILE_DIR/cpu_${PROFILE_ID}.prof"
MEM_PROF="$PROFILE_DIR/mem_${PROFILE_ID}.prof"
ALLOC_PROF="$PROFILE_DIR/alloc_${PROFILE_ID}.prof"

log "Starting profiling run $PROFILE_ID"
log "Model: $MODEL"
log "Engine: ollamarunner (native Go, OLLAMA_NEW_ENGINE=$ENGINE)"
log "Profile directory: $PROFILE_DIR"

# Check if ollama is already running
if pgrep -x ollama >/dev/null; then
	warn "Ollama server already running. Please stop it first:"
	warn "  pkill ollama"
	exit 1
fi

# Start ollama server with profiling enabled
log "Starting Ollama server with profiling enabled..."
OLLAMA_NEW_ENGINE="$ENGINE" \
	OLLAMA_DEBUG=1 \
	OLLAMA_PPROF_DIR="$PROFILE_DIR" \
	ollama serve &
SERVER_PID=$!

# Wait for server to be ready
log "Waiting for server to be ready..."
for i in {1..30}; do
	if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
		log "Server ready"
		break
	fi
	if [[ $i -eq 30 ]]; then
		error "Server failed to start within 30 seconds"
		kill $SERVER_PID 2>/dev/null || true
		exit 1
	fi
	sleep 1
done

# Function to cleanup on exit
cleanup() {
	log "Cleaning up..."
	if [[ -n "${SERVER_PID:-}" ]]; then
		kill $SERVER_PID 2>/dev/null || true
		wait $SERVER_PID 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

# Pull model if not present
log "Ensuring model is available..."
ollama pull "$MODEL" || {
	error "Failed to pull model $MODEL"
	exit 1
}

# Warm up
log "Warming up model..."
echo "Hello" | ollama run "$MODEL" >/dev/null || {
	warn "Warm-up failed (non-critical)"
}

# Run inference with profiling (OLLAMA_PPROF_DIR handles profiling automatically)
log "Running profiled inference..."
START_TIME=$(date +%s)
echo "$PROMPT" | ollama run "$MODEL" >/dev/null || {
	error "Inference failed"
	exit 1
}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "Inference completed in ${DURATION}s"

# The runner writes profiles automatically to OLLAMA_PPROF_DIR
# Find the generated profiles
LATEST_CPU=$(ls -t "$PROFILE_DIR"/cpu_*.prof 2>/dev/null | head -1)
LATEST_MEM=$(ls -t "$PROFILE_DIR"/mem_*.prof 2>/dev/null | head -1)

if [[ ! -f "$LATEST_CPU" ]]; then
	error "No CPU profile found. Check OLLAMA_PPROF_DIR is set correctly."
	exit 1
fi

CPU_PROF="$LATEST_CPU"
MEM_PROF="$LATEST_MEM"
ALLOC_PROF="$MEM_PROF" # Same file, different pprof flags

# Generate profile reports
log "Generating profile reports..."

# Extract profile ID from filename
PROF_ID=$(basename "$CPU_PROF" | sed 's/cpu_\([0-9]*\)\.prof/\1/')

# CPU profile top functions
log "CPU Profile (top 20 functions):"
go tool pprof -top -nodecount=20 "$CPU_PROF" | tee "$PROFILE_DIR/cpu_${PROF_ID}_top.txt"

# Memory profile (in-use)
log "Memory Profile (in-use, top 20):"
go tool pprof -top -nodecount=20 "$MEM_PROF" | tee "$PROFILE_DIR/mem_${PROF_ID}_top.txt"

# Allocation profile
log "Allocation Profile (total allocs, top 20):"
go tool pprof -top -nodecount=20 -alloc_space "$ALLOC_PROF" | tee "$PROFILE_DIR/alloc_${PROF_ID}_top.txt"

# Generate flamegraph-friendly output
if command -v go-torch &>/dev/null; then
	log "Generating flamegraph..."
	go-torch --file="$PROFILE_DIR/cpu_${PROF_ID}_flame.svg" "$CPU_PROF" || warn "Flamegraph generation failed"
fi

# Summary
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Profiling completed successfully"
log "Duration: ${DURATION}s"
log "Profile ID: $PROF_ID"
log "Profiles saved:"
log "  CPU:         $CPU_PROF"
log "  Memory:      $MEM_PROF"
log "  Allocations: $ALLOC_PROF"
log ""
log "View profiles interactively:"
log "  go tool pprof -http=:8080 $CPU_PROF"
log "  go tool pprof -http=:8080 $MEM_PROF"
log "  go tool pprof -http=:8080 -alloc_space $ALLOC_PROF"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
