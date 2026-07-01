package main

import (
	"fmt"
	"net/http"
	"strconv"
)

func getOperands(w http.ResponseWriter, r *http.Request) (float64, float64, bool) {
	aStr := r.URL.Query().Get("a")
	bStr := r.URL.Query().Get("b")

	a, err1 := strconv.ParseFloat(aStr, 64)
	b, err2 := strconv.ParseFloat(bStr, 64)

	if err1 != nil || err2 != nil {
		http.Error(w, "Invalid operands. Use ?a=number&b=number", http.StatusBadRequest)
		return 0, 0, false
	}
	return a, b, true
}

func main() {
	http.HandleFunc("/add", func(w http.ResponseWriter, r *http.Request) {
		a, b, ok := getOperands(w, r)
		if !ok {
			return
		}
		fmt.Fprintf(w, "%.2f\n", a+b)
	})

	http.HandleFunc("/min", func(w http.ResponseWriter, r *http.Request) {
		a, b, ok := getOperands(w, r)
		if !ok {
			return
		}
		fmt.Fprintf(w, "%.2f\n", a-b)
	})

	http.HandleFunc("/mul", func(w http.ResponseWriter, r *http.Request) {
		a, b, ok := getOperands(w, r)
		if !ok {
			return
		}
		fmt.Fprintf(w, "%.2f\n", a*b)
	})

	http.HandleFunc("/div", func(w http.ResponseWriter, r *http.Request) {
		a, b, ok := getOperands(w, r)
		if !ok {
			return
		}
		if b == 0 {
			http.Error(w, "Division by zero is not allowed", http.StatusBadRequest)
			return
		}
		fmt.Fprintf(w, "%.2f\n", a/b)
	})

	fmt.Println("Servidor HTTP da Calculadora executando em http://localhost:8080")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		panic(err)
	}
}
