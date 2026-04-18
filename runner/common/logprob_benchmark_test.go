package common

import (
	"fmt"
	"runtime"
	"testing"
)

// BenchmarkCalculateLogprobs tests OPT-9: sync.Pool buffer reuse
func BenchmarkCalculateLogprobs(b *testing.B) {
	const vocabSize = 151936 // qwen2.5-coder vocab size
	const tokensPerRequest = 421

	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = float32(i % 1000)
	}

	decoder := func(tokenID int) string {
		return fmt.Sprintf("token_%d", tokenID)
	}

	b.Run("WithoutTopK", func(b *testing.B) {
		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				_ = CalculateLogprobs(logits, t%vocabSize, 0, decoder)
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})

	b.Run("WithTopK", func(b *testing.B) {
		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				_ = CalculateLogprobs(logits, t%vocabSize, 5, decoder)
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})
}
