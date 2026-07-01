package main

import "fmt"

var ch chan int

func GoRoutineA() {
	ch <- 1
}
func GoRoutineB() {
	ch <- 1
}
func main() {
	ch = make(chan int)

	go GoRoutineA()
	go GoRoutineB()

	fmt.Scanln()
	// <-ch // blocks forever
}
