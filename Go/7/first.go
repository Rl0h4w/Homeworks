package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type PingResponse struct {
	Message string "json:\"message\""
}

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, world!")
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	response := PingResponse{Message: "pong"}
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		return
	}
}

func main() {
	http.HandleFunc("/hello", helloHandler)
	http.HandleFunc("/ping", pingHandler)
	fmt.Println("Server started 8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println("Error starting server:", err)
	}
}
