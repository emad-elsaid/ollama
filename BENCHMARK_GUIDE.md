# Ollama Native Engine — Benchmark & Profiling Guide

Comprehensive suite for benchmarking and profiling the optimizations documented in 
[OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md).

---

## Quick Start

### End-to-End Tokens/Second Benchmark

```bash
# Run full inference benchmark with default settings
./scripts/benchmark_tokps.sh

# Custom configuration
MODEL=llama3.2:1b BENCHMARK_RUNS=5 WARMUP_RUNS=2 ./scripts/benchmark_tokps.sh

# With custom prompt
PROMPT="Explain quicksort" ./scripts/benchmark_tokps.sh
```

**Example output:**
```
Run 1: 26.57 tok/s
Run 2: 23.21 tok/s
Average: 24.89 tok/s
```

### Run Component Benchmarks

```bash
# Sampling optimizations (OPT-2, OPT-3)
go test -bench=BenchmarkSamplerBufferReuse ./sample -benchmem -benchtime=10s
go test -bench=BenchmarkTopKImplementation ./sample -benchmem -benchtime=10s

# Tensor operations (OPT-4, OPT-6)
go test -bench=BenchmarkTensorFloatsInto ./ml/backend/ggml -benchmem -benchtime=5s
go test -bench=BenchmarkTensorPool ./ml/backend/ggml -benchmem -benchtime=5s

# KV cache (OPT-5)
go test -bench=BenchmarkKVCacheBuildMask ./kvcache -benchmem -benchtime=10s
```

---

## End-to-End Inference Benchmark

### `scripts/benchmark_tokps.sh` — Tokens/Second Measurement

Measures actual model inference throughput (tok/s) across multiple runs.

**Environment variables:**
- `MODEL`: Model to benchmark (default: `qwen2.5-coder:1.5b`)
- `PROMPT`: Prompt for generation (default: bubble sort explanation)
- `WARMUP_RUNS`: Number of warm-up runs (default: `1`)
- `BENCHMARK_RUNS`: Number of benchmark runs (default: `3`)

**Usage:**

```bash
# Default benchmark
./scripts/benchmark_tokps.sh

# Quick single-run test
BENCHMARK_RUNS=1 WARMUP_RUNS=0 ./scripts/benchmark_tokps.sh

# Thorough 10-run benchmark
BENCHMARK_RUNS=10 WARMUP_RUNS=2 ./scripts/benchmark_tokps.sh

# Different model
MODEL=llama3.2:1b ./scripts/benchmark_tokps.sh
```

**Metrics reported:**
- Tokens generated per run
- Generation duration
- Tokens/second throughput
- Average tok/s across all runs

**Baseline (qwen2.5-coder:1.5b on CPU):**
- ~24-27 tok/s on Intel i7-1185G7 @ 3.00GHz

---

## Component Benchmarks

### 1. Sampling Benchmarks (`sample/samplers_benchmark_test.go`)

#### `BenchmarkSamplerBufferReuse` — OPT-2

**Tests:** `tokenBuf` buffer reuse across `Sample()` calls  
**Expected:** 457 MB → 1.3 MB per request (−99.7%)

```bash
go test -bench=BenchmarkSamplerBufferReuse ./sample -benchmem -benchtime=10s
```

**Interpreting results:**
- Look for `MB/request` metric in output
- Before optimization: ~457 MB/request
- After optimization: ~1.3 MB/request
- `B/op` should be minimal (buffer reused, not reallocated)

#### `BenchmarkTopKImplementation` — OPT-3

**Tests:** `slices.SortFunc` vs old `heap.Pop` approach  
**Expected:** 65,536 interface boxing allocations eliminated

```bash
go test -bench=BenchmarkTopKImplementation ./sample -benchmem -benchtime=10s
```

**Interpreting results:**
- Check `allocs/op` — should be near zero
- Before: 65,536 allocations per request
- After: 0 allocations (pdqsort is zero-alloc)

---

### 2. Tensor Operation Benchmarks (`ml/backend/ggml/ggml_benchmark_test.go`)

#### `BenchmarkTensorFloatsInto` — OPT-4

**Tests:** `FloatsInto(dst)` buffer reuse vs old `BackendGet()` allocation  
**Expected:** 226 MB → 0 MB per request (−100%)

```bash
go test -bench=BenchmarkTensorFloatsInto ./ml/backend/ggml -benchmem -benchtime=5s
```

**Interpreting results:**
- Compare `WithReuse` vs `WithoutReuse` sub-benchmarks
- `WithReuse` MB/request should be near zero
- `WithoutReuse` should show ~226 MB/request

#### `BenchmarkTensorPool` — OPT-6

**Tests:** `sync.Pool` recycling of `*Tensor` wrapper objects  
**Expected:** 1.3M objects per request → recycled via pool

```bash
go test -bench=BenchmarkTensorPool ./ml/backend/ggml -benchmem -benchtime=5s
```

**Interpreting results:**
- Check `M_objects/request` metric
- `WithPool`: significantly reduced allocations
- `WithoutPool`: ~1.3M allocations per request baseline

---

### 3. KV Cache Benchmark (`kvcache/causal_benchmark_test.go`)

#### `BenchmarkKVCacheBuildMask` — OPT-5

**Tests:** `maskBuf` reuse in `buildMask()`  
**Expected:** 18 MB → 16 MB per request (initial growth only)

```bash
go test -bench=BenchmarkKVCacheBuildMask ./kvcache -benchmem -benchtime=10s
```

**Interpreting results:**
- `WithReuse`: ~16 MB (one-time growth, then reused)
- `WithoutReuse`: ~18 MB (fresh allocation every forward pass)
- Small delta reflects initial buffer growth cost

---

## End-to-End Profiling

### Profiling Script

The `scripts/profile_optimizations.sh` script automates:
1. Starting Ollama server with profiling enabled
2. Running model inference
3. Capturing CPU, heap, and allocation profiles
4. Generating summary reports

**Usage:**

```bash
# Default: qwen2.5-coder:1.5b
./scripts/profile_optimizations.sh

# Custom model
MODEL=llama3.2:1b ./scripts/profile_optimizations.sh

# Custom prompt
PROMPT="Explain merge sort" ./scripts/profile_optimizations.sh
```

**Environment variables:**
- `MODEL`: Model to profile (default: `qwen2.5-coder:1.5b`)
- `PROMPT`: Inference prompt (default: bubble sort explanation)
- `PROFILE_DIR`: Output directory (default: `ollama-profiles`)
- `OLLAMA_NEW_ENGINE`: Use native engine (default: `1`)

### Analyzing Profiles

#### CPU Profile

```bash
# Top CPU consumers
go tool pprof -top ollama-profiles/cpu_0001.prof

# Interactive web UI
go tool pprof -http=:8080 ollama-profiles/cpu_0001.prof
```

**Key metrics from OPTIMIZATION_PLAN.md:**
- `ggml_vec_dot_q4_K_q8_K_generic`: 60.47% (scalar fallback — OPT-1 targets this)
- `runtime.cgocall`: 23.67% (unavoidable for GGML backend)

**Expected after OPT-1 (SIMD):**  
The `_generic` functions should be replaced by `_avx2` or `_avx512` variants with 
dramatically lower CPU time.

#### Memory Profile (In-Use)

```bash
# Heap in-use at end of request
go tool pprof -top ollama-profiles/mem_0001.prof
```

**After optimizations:**
- `sample.(*Sampler).Sample`: 1.3 MB (was 457 MB)
- `ggml.(*Tensor).Floats`: 0 MB (was 226 MB)
- `kvcache.(*Causal).buildMask`: ~16 MB (was 18 MB)

#### Allocation Profile (Total Bytes)

```bash
# Total bytes allocated over request lifetime
go tool pprof -top -alloc_space ollama-profiles/alloc_0001.prof
```

**Expected reduction:**  
803 MB → 116 MB per request (−86% as documented in OPTIMIZATION_PLAN.md)

---

## Interpreting Benchmark Output

### Standard Go Benchmark Output

```
BenchmarkSamplerBufferReuse/MultiToken-16    50    238475623 ns/op    1.29 MB/request    1349872 B/op    450 allocs/op
```

| Field | Meaning |
|-------|---------|
| `50` | Number of iterations |
| `238475623 ns/op` | Nanoseconds per operation |
| `1.29 MB/request` | **Custom metric**: Total MB allocated per 421-token request |
| `1349872 B/op` | Bytes allocated per benchmark iteration |
| `450 allocs/op` | Allocation count per iteration |

### Custom Metrics

Our benchmarks report domain-specific metrics:

- **`MB/request`**: Total memory allocated for a full 421-token generation (from profiling data)
- **`M_objects/request`**: Millions of objects allocated per request

These map directly to the metrics in `OPTIMIZATION_PLAN.md`.

---

## Regression Testing

### Automated Benchmark Comparison

```bash
# Baseline (before changes)
go test -bench=. ./sample ./kvcache ./ml/backend/ggml -benchmem -benchtime=10s > bench_before.txt

# After changes
go test -bench=. ./sample ./kvcache ./ml/backend/ggml -benchmem -benchtime=10s > bench_after.txt

# Compare
go install golang.org/x/perf/cmd/benchstat@latest
benchstat bench_before.txt bench_after.txt
```

**Example output:**
```
name                               old MB/request    new MB/request    delta
SamplerBufferReuse/MultiToken      457.2 ± 2%        1.3 ± 1%          -99.72%
```

---

## Continuous Monitoring

### GitHub Actions Benchmark

Add to `.github/workflows/benchmark.yml`:

```yaml
name: Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.23'
      
      - name: Run benchmarks
        run: |
          go test -bench=BenchmarkSamplerBufferReuse ./sample -benchmem
          go test -bench=BenchmarkKVCacheBuildMask ./kvcache -benchmem
          go test -bench=BenchmarkTensorFloatsInto ./ml/backend/ggml -benchmem
```

---

## Troubleshooting

### GGML Backend Benchmarks Fail

**Issue:** `no CPU backend available`

**Solution:** Benchmarks require a built GGML backend. Run:

```bash
cmake -B build -S . && cmake --build build --config Release
```

### Profiling Script Can't Connect

**Issue:** `Server failed to start within 30 seconds`

**Solutions:**
- Ensure no other Ollama instance is running: `pkill ollama`
- Check port 11434 is available: `lsof -i :11434`
- Increase timeout in script (line 65)

### OPT-1 (SIMD) Not Showing in Profiles

**Issue:** Still seeing `_generic` functions in CPU profile

**Cause:** CMake change requires full rebuild

**Solution:**
```bash
rm -rf build
cmake -B build -S . && cmake --build build --config Release
```

Verify with:
```bash
nm build/lib/libggml.so | grep ggml_vec_dot_q4_K_q8_K
# Should show _avx2 or _avx512 symbols, not just _generic
```

---

## References

- [OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md) — Full optimization analysis
- [Go Profiling Guide](https://go.dev/blog/pprof)
- [Benchstat Documentation](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat)
