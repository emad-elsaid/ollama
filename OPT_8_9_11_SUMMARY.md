# OPT-8, OPT-9, OPT-11 Implementation Summary

**Date**: 2026-04-19  
**Commit**: Implementation of high-priority buffer reuse optimizations

---

## Optimizations Implemented

### OPT-9: Logprob Buffer Reuse ✅ **Completed**

**Files Modified**: `runner/common/logprob.go`

**Changes**:
- Added `sync.Pool` for reusable `logprobBuffers` struct
- Pre-allocated buffers with 200,000 capacity for common vocab sizes
- Eliminated per-token allocations of:
  - `logProbs []float32` (600 KB for qwen2.5-coder)
  - `pairs []tokenLogprobPair` (1.2 MB for qwen2.5-coder)

**Benchmark Results**:
```
WithoutTopK:  0.829 MB/request   829 KB/op    1,009 allocs/op
WithTopK:     4.179 MB/request   4.2 MB/op    6,917 allocs/op
```

**Impact Analysis**:
- **Without TopK**: 0.829 MB vs baseline ~252 MB (99.67% reduction)
- **With TopK (typical)**: 4.179 MB vs baseline ~758 MB (99.45% reduction)
- Only TopLogprob result slice (5-10 tokens) still allocated per call
- Buffer growth happens once, then reused across all subsequent calls

**Validation**: ✅ All tests pass (`TestCalculateLogprobs*`)

---

### OPT-8: computeBatch Buffer Reuse ✅ **Completed**

**Files Modified**: `runner/ollamarunner/runner.go`

**Changes**:
- Added three reusable buffers to `Server` struct:
  - `batchInputsBuf []int32`
  - `nextTokensBuf []*input.Input`
  - `iBatchesBuf []int`
- Implemented grow-as-needed pattern matching OPT-4 (`logitsBuf`)
- Added explicit nil-clearing for `nextTokensBuf` to prevent stale pointers

**Code Pattern**:
```go
if cap(s.batchInputsBuf) < len(activeBatch.batchInputs) {
    s.batchInputsBuf = make([]int32, len(activeBatch.batchInputs))
}
batchInputs := s.batchInputsBuf[:len(activeBatch.batchInputs)]
```

**Impact Analysis**:
- **Frequency**: 3 allocations eliminated per batch
- **Size**: Varies by batch size and parallel load
  - Typical: 1-16 KB per batch (parallel=1-4)
  - High load: up to 64 KB per batch (parallel=16)
- **GC Pressure**: Reduced by consistent small wins

**Validation**: ✅ Package compiles, no existing tests for this path

---

### OPT-11: Grammar Sampler Buffer Reuse ✅ **Completed**

**Files Modified**: `sample/samplers.go`

**Changes**:
- Added `tokenDataBuf []llama.TokenData` field to `GrammarSampler`
- Reuses buffer in `Apply()` method (called per token when grammar enabled)
- Eliminates allocation of vocab_size × 16 bytes per token

**Code Pattern**:
```go
if cap(g.tokenDataBuf) < len(tokens) {
    g.tokenDataBuf = make([]llama.TokenData, len(tokens))
}
tds := g.tokenDataBuf[:len(tokens)]
```

**Impact Analysis**:
- **Frequency**: Per-token when constrained generation used
- **Size**: ~2.4 MB per token (qwen2.5-coder: 151,936 tokens × 16 bytes)
- **Total Saved**: ~1 GB per 421-token request (when grammar enabled)
- **Baseline Allocation**: 2.4 MB × 421 = 1,010 MB

**Validation**: ✅ `TestGrammar` passes

---

## Estimated Impact Summary

| Optimization | When Active | Per-Request Savings | Reduction |
|--------------|-------------|---------------------|-----------|
| **OPT-9** (Logprobs) | When logprobs=true | 758 MB → 4 MB | **99.45%** |
| **OPT-8** (computeBatch) | Always | ~5-50 KB | Minor GC |
| **OPT-11** (Grammar) | When grammar set | 1,010 MB → ~2.4 MB | **99.76%** |

**Combined with OPT-2, OPT-4, OPT-5, OPT-6**:
- Total allocation reduction: **~90-95%** in typical scenarios
- Logprobs enabled: **~95-98%** reduction
- Grammar enabled: **~98-99%** reduction

---

## Testing Strategy

**Unit Tests**: ✅ Pass
- `TestCalculateLogprobs*` (all 7 test cases)
- `TestGrammar`

**Compilation**: ✅ Success
- `runner/common` package
- `runner/ollamarunner` package
- `sample` package

**Benchmark Validation**: ✅ Confirmed
- OPT-9 benchmark shows 99.45% reduction with TopK
- Buffers properly reused via sync.Pool

**Integration Testing Recommended**:
- End-to-end generation with logprobs enabled
- Grammar-constrained generation
- High parallel load scenarios

---

## Code Quality

**Patterns Used**:
- Consistent with existing OPT-4 (logitsBuf) pattern
- Proper nil-clearing for pointer slices (prevents stale references)
- sync.Pool for zero-API-impact optimization
- Grow-as-needed: buffers expand once, never shrink

**Safety**:
- Thread-safe (sync.Pool handles concurrency)
- No behavioral changes (buffers transparent to callers)
- Preserves existing semantics completely

**Maintainability**:
- Well-commented with OPT-X references
- Minimal code delta (high value/complexity ratio)
- No new dependencies

---

## Next Steps

1. ✅ Verify changes compile
2. ✅ Run unit tests
3. ✅ Benchmark OPT-9
4. ⏳ Run end-to-end tok/s benchmark
5. ⏳ Update `OPTIMIZATION_PLAN.md` with results

---

## Files Changed

- `runner/common/logprob.go` — OPT-9 implementation
- `runner/common/logprob_benchmark_test.go` — OPT-9 benchmark (new file)
- `runner/ollamarunner/runner.go` — OPT-8 implementation
- `sample/samplers.go` — OPT-11 implementation
- `.gitignore` — Added `*.prof` and `*.test` patterns

**Total Lines Modified**: ~50 lines across 4 files
