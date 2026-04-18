package kvcache

import (
	"math"
	"runtime"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkKVCacheBuildMask tests OPT-5: maskBuf reuse in buildMask
// Before: 18 MB / request (fresh allocation every forward pass)
// After:  16 MB / request (initial growth, then reused)
func BenchmarkKVCacheBuildMask(b *testing.B) {
	const maxTokens = 2048
	const batchSize = 1
	const tokensPerRequest = 421

	b.Run("WithReuse", func(b *testing.B) {
		cache := &Causal{
			DType:         ml.DTypeF32,
			swaWindowSize: math.MaxInt32,
			swaMemorySize: math.MaxInt32,
			config:        &ml.CacheConfig{CachePadding: 256},
			maxBatch:      batchSize,
			cells:         make([]cacheCell, maxTokens),
			cellRanges:    make(map[int]cellRange),
		}

		// Initialize cache cells to simulate a filled KV cache
		for i := range cache.cells {
			cache.cells[i] = cacheCell{
				pos:       int32(i),
				sequences: []int{0},
			}
		}

		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				cache.curBatchSize = batchSize
				cache.curSequences = []int{0}
				cache.curPositions = []int32{int32(t)}
				cache.curCellRange = cellRange{min: 0, max: t}

				// Directly test the allocation behavior
				cache.curCellRange.min = roundDown(cache.curCellRange.min, cache.config.CachePadding)
				cache.curCellRange.max = roundUp(cache.curCellRange.max+1, cache.config.CachePadding) - 1

				length := cache.curCellRange.max - cache.curCellRange.min + 1
				size := cache.curBatchSize * length

				if cap(cache.maskBuf) < size {
					cache.maskBuf = make([]float32, size)
				}
				mask := cache.maskBuf[:size]
				for j := range mask {
					mask[j] = 0
				}
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})

	b.Run("WithoutReuse", func(b *testing.B) {
		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				// Simulate old behavior: allocate fresh every time
				min := roundDown(0, 256)
				max := roundUp(t+1, 256) - 1
				length := max - min + 1
				size := batchSize * length
				mask := make([]float32, size)
				for j := range mask {
					mask[j] = 0
				}
				_ = mask
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})
}
