package main

import (
	"fmt"
	"io"
	"net/http"
)

func callOperation(op string, a, b float64) {
	url := fmt.Sprintf("http://localhost:8080/%s?a=%f&b=%f", op, a, b)

	resp, err := http.Get(url)
	if err != nil {
		fmt.Printf("Error calling %s: %v\n", op, err)
		return
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Error reading response from %s: %v\n", op, err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("%s failed: %s\n", op, string(body))
		return
	}

	fmt.Printf("%s(%v, %v) = %s", op, a, b, string(body))
}

func main() {
	callOperation("add", 10, 5)
	callOperation("min", 10, 5)
	callOperation("mul", 7, 6)
	callOperation("div", 8, 2)
}
