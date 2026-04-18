package ggml

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkTensorFloatsInto tests OPT-4: FloatsInto buffer reuse
// Before: 226 MB / request (600 KB × 421 tokens)
// After:  0 MB / request (logitsBuf reused)
func BenchmarkTensorFloatsInto(b *testing.B) {
	const vocabSize = 151936 // qwen2.5-coder vocab size
	const tokensPerRequest = 421

	ctx := setup(b)

	tensor := ctx.Empty(ml.DTypeF32, vocabSize)
	// Populate with test data
	data := make([]float32, vocabSize)
	for i := range data {
		data[i] = float32(i % 1000)
	}
	tensor.FromFloats(data)

	b.Run("WithReuse", func(b *testing.B) {
		b.ReportAllocs()
		reusableBuf := make([]float32, 0, vocabSize)

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			for t := 0; t < tokensPerRequest; t++ {
				reusableBuf = tensor.FloatsInto(reusableBuf)
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
				_ = tensor.BackendGet() // Old method: always allocates
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		allocsPerRequest := float64(m2.TotalAlloc-m1.TotalAlloc) / float64(b.N)
		b.ReportMetric(allocsPerRequest/1e6, "MB/request")
	})
}

// BenchmarkTensorPool tests OPT-6: sync.Pool for *Tensor wrapper allocations
// Before: 1.3M wrapper objects / request
// After:  Wrappers recycled via pool
func BenchmarkTensorPool(b *testing.B) {
	const layerOps = 100 // approximate tensor ops per layer
	const layers = 28    // qwen2.5-coder:1.5b has 28 layers
	const tokensPerRequest = 421

	totalOpsPerRequest := layerOps * layers * tokensPerRequest

	b.Run("WithPool", func(b *testing.B) {
		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			ctx := setup(b)

			// Simulate tensor operations that use poolTensor
			for op := 0; op < totalOpsPerRequest; op++ {
				t := ctx.Empty(ml.DTypeF32, 128)
				_ = t
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		objectsPerRequest := float64(m2.Mallocs-m1.Mallocs) / float64(b.N)
		b.ReportMetric(objectsPerRequest/1e6, "M_objects/request")
	})

	b.Run("WithoutPool", func(b *testing.B) {
		b.ReportAllocs()

		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			// Simulate old behavior: direct allocation without pooling
			for op := 0; op < totalOpsPerRequest; op++ {
				t := &Tensor{}
				_ = t
			}
		}

		b.StopTimer()
		runtime.ReadMemStats(&m2)
		objectsPerRequest := float64(m2.Mallocs-m1.Mallocs) / float64(b.N)
		b.ReportMetric(objectsPerRequest/1e6, "M_objects/request")
	})
}
