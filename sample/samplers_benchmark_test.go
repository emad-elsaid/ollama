package sample

import (
	"fmt"
	"math/rand"
	"runtime"
	"slices"
	"testing"
)

func BenchmarkWeightedSampler(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5)
			}

			sampler := NewSampler(0.8, 0, 0, 0, 42, nil)
			b.ResetTimer()
			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}

	configs := []struct {
		name        string
		temperature float32
		topK        int
		topP        float32
		minP        float32
		seed        int
	}{
		{"Greedy", 0, -1, 0, 0, -1},
		{"Temperature", 0.8, -1, 0, 0, -1},
		{"TopK", 0.8, 50, 0, 0, -1},
		{"TopP", 0.8, -1, 0.9, 0, -1},
		{"MinP", 0.8, -1, 0, 0.05, -1},
		{"WithSeed", 0.8, 50, 0, 0, 42},
	}

	// Fixed size for common vocab size
	size := 128000
	logits := make([]float32, size)
	for i := range logits {
		logits[i] = float32(rand.Float64()*10 - 5)
	}

	for _, tc := range configs {
		b.Run("Config"+tc.name, func(b *testing.B) {
			sampler := NewSampler(tc.temperature, tc.topK, tc.topP, tc.minP, tc.seed, nil)
			sampler.Sample(logits)

			b.ResetTimer()

			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}

	// Test with combined transforms separately - topK influences performance greatly
	b.Run("TransformCombined", func(b *testing.B) {
		sampler := NewSampler(0.8, 50, 0.9, 0.05, 42, nil)
		b.ResetTimer()

		for b.Loop() {
			sampler.Sample(logits)
		}
	})
}

func BenchmarkGreedySampler(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5)
			}

			sampler := NewSampler(0, -1, 0, 0, -1, nil)
			b.ResetTimer()

			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}
}

// BenchmarkSamplerBufferReuse tests OPT-2: tokenBuf reuse across Sample calls
// Before: 457 MB / request (1.16 MB × 421 tokens)
// After:  1.3 MB / request (buffer grown once, reused)
func BenchmarkSamplerBufferReuse(b *testing.B) {
	const vocabSize = 151936 // qwen2.5-coder vocab size from profiling
	const tokensPerRequest = 421

	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = float32(rand.Float64()*10 - 5)
	}

	sampler := NewSampler(0.8, 50, 0.9, 0, 42, nil)

	b.Run("MultiToken", func(b *testing.B) {
		b.ReportAllocs()
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				sampler.Sample(logits)
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})
}

// BenchmarkTopKImplementation tests OPT-3: slices.SortFunc vs heap.Pop
// Before: 65,536 interface boxing allocations via heap.Pop
// After:  0 allocations via slices.SortFunc
func BenchmarkTopKImplementation(b *testing.B) {
	const vocabSize = 151936
	const k = 50 // typical topK value

	tokens := make([]token, vocabSize)
	for i := range tokens {
		tokens[i].id = int32(i)
		tokens[i].value = float32(rand.Float64()*10 - 5)
	}

	b.Run("CurrentSortFunc", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			result := topK(slices.Clone(tokens), k)
			_ = result
		}
	})
}
