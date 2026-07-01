package main

import "fmt"

var x int

func GoroutineA() {
	x = 3
}

func GoroutineB() {
	x = 4
}
func main() {
	go GoroutineA()
	go GoroutineB()

	fmt.Println(x)

	fmt.Scanln()
}
