package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

const N = 20
const SampleSize = 30
const GoMaxProcs = 6

func main() {
	runtime.GOMAXPROCS(GoMaxProcs)
	wg := sync.WaitGroup{}

	println("CPUs Lógicas:", runtime.NumCPU())

	t1 := time.Now()
	for i := 0; i < SampleSize; i++ {
		wg.Add(1)
		go task(&wg)
	}
	wg.Wait()
	fmt.Println(float64(time.Now().Sub(t1).Milliseconds())/float64(SampleSize), "ms")
}

func task(wg *sync.WaitGroup) {
	defer wg.Done()
	for i := 0; i < 10000; i++ {
		fibonacciRecursive(N)
	}
}

func fibonacciRecursive(n int) int {
	if n <= 1 {
		return n
	}
	return fibonacciRecursive(n-1) + fibonacciRecursive(n-2)
}
