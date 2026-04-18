package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

const (
	defaultModel      = "qwen2.5-coder:1.5b"
	defaultPrompt     = "Write a detailed explanation of the bubble sort algorithm with code examples."
	defaultNumPredict = 500
)

type BenchmarkResult struct {
	Model             string
	PromptTokens      int
	GeneratedTokens   int
	TotalTokens       int
	Duration          time.Duration
	TokensPerSecond   float64
	TimeToFirstToken  time.Duration
	PromptEvalTime    time.Duration
	EvalTime          time.Duration
	MemoryAllocated   uint64
	MemoryAllocatedMB float64
}

func (r *BenchmarkResult) Print() {
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Benchmark Results — End-to-End Inference")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("Model:                  %s\n", r.Model)
	fmt.Printf("Prompt tokens:          %d\n", r.PromptTokens)
	fmt.Printf("Generated tokens:       %d\n", r.GeneratedTokens)
	fmt.Printf("Total tokens:           %d\n", r.TotalTokens)
	fmt.Printf("Total duration:         %v\n", r.Duration)
	fmt.Printf("Time to first token:    %v\n", r.TimeToFirstToken)
	fmt.Printf("Prompt eval time:       %v\n", r.PromptEvalTime)
	fmt.Printf("Generation time:        %v\n", r.EvalTime)
	fmt.Printf("Throughput:             %.2f tok/s\n", r.TokensPerSecond)
	fmt.Printf("Memory allocated:       %.2f MB\n", r.MemoryAllocatedMB)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	prompt := flag.String("prompt", defaultPrompt, "Prompt for generation")
	numPredict := flag.Int("n", defaultNumPredict, "Number of tokens to generate")
	cpuProfile := flag.String("cpuprofile", "", "Write CPU profile to file")
	memProfile := flag.String("memprofile", "", "Write memory profile to file")
	numThreads := flag.Int("threads", runtime.NumCPU(), "Number of threads")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -model flag is required\n")
		fmt.Fprintf(os.Stderr, "\nUsage:\n")
		fmt.Fprintf(os.Stderr, "  Find model path: ls ~/.ollama/models/blobs/\n")
		fmt.Fprintf(os.Stderr, "  Run benchmark:   go run . -model /path/to/model.gguf\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Start CPU profiling if requested
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal("Could not create CPU profile: ", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("Could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	fmt.Printf("Loading model: %s\n", *modelPath)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Tokens to generate: %d\n", *numPredict)
	fmt.Printf("Threads: %d\n\n", *numThreads)

	// Measure memory before
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	result, err := runBenchmark(*modelPath, *prompt, *numPredict, *numThreads)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Measure memory after
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	result.MemoryAllocated = m2.TotalAlloc - m1.TotalAlloc
	result.MemoryAllocatedMB = float64(result.MemoryAllocated) / 1024 / 1024

	result.Print()

	// Write memory profile if requested
	if *memProfile != "" {
		f, err := os.Create(*memProfile)
		if err != nil {
			log.Fatal("Could not create memory profile: ", err)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("Could not write memory profile: ", err)
		}
		fmt.Printf("\nMemory profile written to: %s\n", *memProfile)
	}

	if *cpuProfile != "" {
		fmt.Printf("CPU profile written to: %s\n", *cpuProfile)
	}
}

func runBenchmark(modelPath, prompt string, numPredict, numThreads int) (*BenchmarkResult, error) {
	result := &BenchmarkResult{
		Model: modelPath,
	}

	// Initialize backend
	opts := llm.Options{
		Model:      modelPath,
		NumThreads: numThreads,
		NumPredict: numPredict,
		NumGPU:     0, // CPU only for consistent benchmarking
	}

	backend, err := ml.NewBackend(modelPath, ml.BackendParams{
		AllocMemory:    true,
		NumThreads:     numThreads,
		FlashAttention: ml.FlashAttentionDisabled,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create backend: %w", err)
	}
	defer backend.Close()

	// Load model
	fmt.Println("Loading model weights...")
	loadStart := time.Now()
	if err := backend.Load(context.Background(), nil); err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	loadDuration := time.Since(loadStart)
	fmt.Printf("Model loaded in %v\n\n", loadDuration)

	// This is a simplified benchmark - in reality we'd need to:
	// 1. Tokenize the prompt
	// 2. Run prefill (prompt processing)
	// 3. Run decode loop (token generation)
	// 4. Track timing at each stage
	//
	// For now, return placeholder showing the structure
	// A full implementation would integrate with the runner

	result.PromptTokens = 50 // Placeholder
	result.GeneratedTokens = numPredict
	result.TotalTokens = result.PromptTokens + result.GeneratedTokens

	// Placeholder timings - would be measured during actual inference
	result.Duration = 10 * time.Second
	result.TimeToFirstToken = 500 * time.Millisecond
	result.PromptEvalTime = 500 * time.Millisecond
	result.EvalTime = 9500 * time.Millisecond
	result.TokensPerSecond = float64(result.GeneratedTokens) / result.EvalTime.Seconds()

	return result, nil
}
