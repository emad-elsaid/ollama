# Optimization Roadmap — Additional Opportunities

**Date**: 2026-04-19  
**Status**: Proposed optimizations beyond OPT-1 through OPT-7

---

## Summary of Findings

Codebase analysis reveals **8 additional optimization opportunities** in hot paths:

**High Priority** (per-token allocations):
- OPT-8: Buffer reuse in computeBatch (3 allocations/batch)
- OPT-9: Logprob calculation buffer reuse (2× vocab_size allocations when enabled)
- OPT-10: String concatenation in response handling

**Medium Priority** (per-batch or conditional):
- OPT-11: Grammar sampler TokenData buffer reuse
- OPT-12: Pre-allocation in forwardBatch loops
- OPT-13: KV cache cell lookup optimization (O(n×m) → O(n))

**Low Priority** (rare cases or already acceptable):
- OPT-14: Stop sequence checking without substring allocations
- OPT-15: Tensor pool lock contention (only under extreme parallelism)

---

## OPT-8: Buffer Reuse in computeBatch ⏳ proposed

**Priority**: HIGH  
**Impact**: 3 allocations per batch (varies by parallel load)  
**Files**: `runner/ollamarunner/runner.go:670-679`

### Current Code

```go
func (s *Server) computeBatch(activeBatch *activeBatch) {
    batchInputs := make([]int32, len(activeBatch.batchInputs))  // Allocation 1
    // ... 
    nextBatchTokens := make([]*input.Input, len(s.seqs))        // Allocation 2
    iBatches := make([]int, len(s.seqs))                        // Allocation 3
```

### Proposed Optimization

Add reusable buffers to `Server` struct:

```go
type Server struct {
    // ... existing fields
    logitsBuf []float32  // Already optimized in OPT-4
    
    // New reusable buffers
    batchInputsBuf []int32
    nextTokensBuf  []*input.Input
    iBatchesBuf    []int
    
    mu sync.Mutex
}

func (s *Server) computeBatch(activeBatch *activeBatch) {
    // Reuse with grow-as-needed pattern
    if cap(s.batchInputsBuf) < len(activeBatch.batchInputs) {
        s.batchInputsBuf = make([]int32, len(activeBatch.batchInputs))
    }
    batchInputs := s.batchInputsBuf[:len(activeBatch.batchInputs)]
    
    if cap(s.nextTokensBuf) < len(s.seqs) {
        s.nextTokensBuf = make([]*input.Input, len(s.seqs))
    }
    nextBatchTokens := s.nextTokensBuf[:len(s.seqs)]
    
    if cap(s.iBatchesBuf) < len(s.seqs) {
        s.iBatchesBuf = make([]int, len(s.seqs))
    }
    iBatches := s.iBatchesBuf[:len(s.seqs)]
```

### Estimated Impact

- **Memory**: ~1-16 KB per batch saved (depends on batch size)
- **Throughput**: Minor improvement (<1%) - reduces GC pressure
- **Complexity**: Low - same pattern as OPT-4

---

## OPT-9: Logprob Buffer Reuse ⏳ proposed

**Priority**: HIGH (when logprobs enabled)  
**Impact**: 2× vocab_size allocations per token  
**Files**: `runner/common/logprob.go:34-67`

### Current Code

```go
func CalculateLogprobs(logits []float32, selectedToken int, topK int, decoder TokenDecoderFunc) []llm.Logprob {
    // ...
    logProbs := make([]float32, len(logits))  // Allocation 1: vocab_size (151,936 × 4 = 600 KB)
    
    if topK > 0 {
        pairs := make([]tokenLogprobPair, len(logProbs))  // Allocation 2: vocab_size × 8 bytes
```

### Issue Analysis

**Frequency**: Called once per generated token when logprobs requested  
**Size**: 
- `logProbs`: 600 KB (qwen2.5-coder vocab)
- `pairs`: 1.2 MB (8 bytes per pair)
- **Total**: 1.8 MB per token

**Over 421 tokens**: 758 MB allocated (comparable to OPT-2 impact)

### Proposed Optimization

**Option A**: Add buffers to Server struct (requires API change)

```go
type LogprobCalculator struct {
    logProbsBuf []float32
    pairsBuf    []tokenLogprobPair
}

func (lc *LogprobCalculator) Calculate(logits []float32, selectedToken int, topK int, decoder TokenDecoderFunc) []llm.Logprob {
    if cap(lc.logProbsBuf) < len(logits) {
        lc.logProbsBuf = make([]float32, len(logits))
    }
    logProbs := lc.logProbsBuf[:len(logits)]
    
    if topK > 0 {
        if cap(lc.pairsBuf) < len(logits) {
            lc.pairsBuf = make([]tokenLogprobPair, len(logits))
        }
        pairs := lc.pairsBuf[:len(logits)]
```

**Option B**: Use sync.Pool (simpler, no API change)

```go
var logprobPool = sync.Pool{
    New: func() any {
        return &logprobBuffers{
            logProbs: make([]float32, 0, 200000),  // Pre-size for common vocabs
            pairs:    make([]tokenLogprobPair, 0, 200000),
        }
    },
}

func CalculateLogprobs(logits []float32, ...) []llm.Logprob {
    bufs := logprobPool.Get().(*logprobBuffers)
    defer logprobPool.Put(bufs)
    
    logProbs := bufs.logProbs[:len(logits)]
    // ... use and return
```

### Estimated Impact

- **Memory**: 758 MB saved per 421-token request (when logprobs enabled)
- **Reduction**: ~100% of logprob allocations
- **Throughput**: 5-10% improvement when logprobs enabled
- **Complexity**: Medium (Option A) / Low (Option B)

---

## OPT-10: String Concatenation Optimization ⏳ proposed

**Priority**: MEDIUM  
**Impact**: Per-token string allocations  
**Files**: `runner/ollamarunner/runner.go:412, 809`

### Current Code

```go
strings.Join(seq.pendingResponses, "")  // Line 412
strings.Join(seq.pendingResponses, "")  // Line 809
```

### Issue

`strings.Join` creates new string allocation each call. Called per-token during generation.

### Proposed Optimization

**Option A**: Use strings.Builder with running accumulator

```go
type sequence struct {
    // Replace []string with strings.Builder
    responseBuilder strings.Builder
    pendingCount    int
}

// In token generation:
seq.responseBuilder.WriteString(piece)
seq.pendingCount++

// When flushing:
fullResponse := seq.responseBuilder.String()
seq.responseBuilder.Reset()
seq.pendingCount = 0
```

**Option B**: Keep slice but optimize join call

```go
// Cache joined result, only rebuild when new tokens added
type sequence struct {
    pendingResponses []string
    cachedJoin       string
    joinDirty        bool
}
```

### Estimated Impact

- **Memory**: ~5-50 KB per request (depends on response length)
- **Throughput**: <1% improvement
- **Complexity**: Medium (Option A), Low (Option B)

---

## OPT-11: Grammar Sampler Buffer Reuse ⏳ proposed

**Priority**: MEDIUM (only when grammar enabled)  
**Impact**: vocab_size × 16 bytes per token  
**Files**: `sample/samplers.go:192-196`

### Current Code

```go
func (s *Sampler) sample(tokens []token) (token, error) {
    if s.grammar != nil {
        tds := make([]llama.TokenData, len(tokens))  // Allocation: ~2.4 MB for qwen2.5
```

### Proposed Optimization

```go
type GrammarSampler struct {
    // ... existing fields
    tokenDataBuf []llama.TokenData
}

func (g *GrammarSampler) Apply(tokens []token) {
    if cap(g.tokenDataBuf) < len(tokens) {
        g.tokenDataBuf = make([]llama.TokenData, len(tokens))
    }
    tds := g.tokenDataBuf[:len(tokens)]
```

### Estimated Impact

- **Memory**: ~1 GB saved per request (when grammar enabled)
- **Frequency**: Only when constrained generation used
- **Complexity**: Low

---

## OPT-12: Pre-allocation in forwardBatch ⏳ proposed

**Priority**: MEDIUM  
**Impact**: Multiple append reallocations per batch  
**Files**: `runner/ollamarunner/runner.go:587-605`

### Current Code

```go
func (s *Server) forwardBatch() *activeBatch {
    var batchInputs []*input.Input
    // ...
    for _, seq := range s.seqs {
        for _, input := range seq.inputs {
            batchInputs = append(batchInputs, input)  // Grows dynamically
            batch.Positions = append(batch.Positions, ...)
            batch.Sequences = append(batch.Sequences, ...)
```

### Proposed Optimization

```go
// Pre-calculate size
totalInputs := 0
for _, seq := range s.seqs {
    if seq != nil {
        totalInputs += len(seq.inputs)
    }
}

// Pre-allocate exact size
batchInputs := make([]*input.Input, 0, totalInputs)
batch.Positions = make([]int32, 0, totalInputs)
batch.Sequences = make([]int, 0, totalInputs)
```

### Estimated Impact

- **Memory**: Eliminates ~3-5 realloc operations per batch
- **Throughput**: <1% improvement
- **Complexity**: Low

---

## OPT-13: KV Cache Cell Lookup Optimization ⏳ proposed

**Priority**: MEDIUM  
**Impact**: O(n×m) → O(n) in sliding window updates  
**Files**: `kvcache/causal.go:312-351`

### Current Code

```go
for seq, seqRange := range c.cellRanges {
    for i := seqRange.min; i <= seqRange.max; i++ {
        if slices.Contains(c.cells[i].sequences, seq) {  // O(m) per iteration
            last = max(last, c.cells[i].pos)
        }
    }
}
```

### Issue

- **Outer loop**: O(sequences)
- **Inner loop**: O(cell_range)
- **Contains check**: O(sequences_per_cell)
- **Total**: O(sequences × cell_range × sequences_per_cell)

For typical cases: sequences=1-4, cell_range=2048, sequences_per_cell=1-2  
**Current**: ~2,048-16,384 operations per update

### Proposed Optimization

Add reverse index:

```go
type Causal struct {
    // ... existing fields
    seqToCells map[int][]int  // seq -> list of cell indices
}

// Update index when modifying cells
func (c *Causal) Put(...) {
    // ... existing code
    if c.seqToCells == nil {
        c.seqToCells = make(map[int][]int)
    }
    c.seqToCells[seq] = append(c.seqToCells[seq], cellIdx)
}

// Fast lookup
for seq := range c.cellRanges {
    cellIndices := c.seqToCells[seq]
    for _, i := range cellIndices {
        last = max(last, c.cells[i].pos)
    }
}
```

### Estimated Impact

- **Complexity**: O(n×m) → O(n)
- **Memory**: ~16-64 KB additional (map overhead)
- **Throughput**: 2-5% improvement in sliding window scenarios
- **Trade-off**: Memory for speed

---

## OPT-14: Stop Sequence Checking ⏳ proposed

**Priority**: LOW  
**Impact**: String allocations in stop detection  
**Files**: `runner/common/stop.go:17-46`

### Current Code

```go
func ContainsStopSuffix(sequence string, stops []string) bool {
    for _, stop := range stops {
        for i := 1; i <= len(stop); i++ {
            if strings.HasSuffix(sequence, stop[:i])  // Creates substring
```

### Proposed Optimization

Use byte-level comparison:

```go
func ContainsStopSuffix(sequence string, stops []string) bool {
    seqBytes := []byte(sequence)
    for _, stop := range stops {
        stopBytes := []byte(stop)
        for i := 1; i <= len(stopBytes); i++ {
            if bytes.HasSuffix(seqBytes, stopBytes[:i])  // No allocation
```

### Estimated Impact

- **Memory**: ~1-10 KB saved per check
- **Frequency**: Once per generated token
- **Complexity**: Low

---

## OPT-15: Tensor Pool Lock Contention ⏳ proposed

**Priority**: LOW (only under extreme parallelism)  
**Impact**: Lock contention with >8 parallel sequences  
**Files**: `ml/backend/ggml/ggml.go:47-56`

### Current Code

```go
var tensorPool = sync.Pool{
    New: func() any { return new(Tensor) },
}
```

### Issue

Single global pool can bottleneck under high concurrency (parallel>16).

### Proposed Optimization

Per-context pools or sharded pools:

```go
const numShards = 16

var tensorPools [numShards]sync.Pool

func getTensorPool(ctx *Context) *sync.Pool {
    shard := uintptr(unsafe.Pointer(ctx)) % numShards
    return &tensorPools[shard]
}
```

### Estimated Impact

- **Frequency**: Only matters with parallel>16
- **Improvement**: 10-20% in extreme concurrency
- **Complexity**: Medium

---

## Prioritized Implementation Order

| # | Optimization | Impact | Complexity | Recommended |
|---|--------------|--------|------------|-------------|
| **1** | OPT-9 (Logprobs) | 758 MB when enabled | Low (Pool) | ✅ Yes |
| **2** | OPT-8 (computeBatch) | Minor GC pressure | Low | ✅ Yes |
| **3** | OPT-11 (Grammar) | 1 GB when enabled | Low | ✅ Yes |
| **4** | OPT-13 (KV lookup) | 2-5% in SWA | Medium | ⚠️ Profile first |
| **5** | OPT-12 (forwardBatch) | <1% | Low | ⚠️ Profile first |
| **6** | OPT-10 (String join) | <1% | Medium | ❌ Low value |
| **7** | OPT-14 (Stop check) | Minimal | Low | ❌ Low value |
| **8** | OPT-15 (Pool sharding) | Only at scale | Medium | ❌ Premature |

---

## Next Steps

1. **Implement OPT-9** (logprobs buffer reuse) - highest impact when enabled
2. **Implement OPT-8** (computeBatch buffers) - consistent small win
3. **Implement OPT-11** (grammar buffers) - conditional but large when triggered
4. **Profile OPT-13** - benchmark sliding window workloads to validate 2-5% claim
5. **Skip OPT-14, OPT-15** - optimization effort > benefit

**Total estimated additional improvement**: 10-15% throughput when logprobs enabled, 1-3% baseline improvement.
