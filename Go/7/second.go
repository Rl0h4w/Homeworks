package main

import (
	"fmt"
	"net/http"
	"time"
)

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func timeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, time.Now())
}

func dateHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, time.Now().Format("02-01-2006"))
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/time", timeHandler)
	mux.HandleFunc("/date", dateHandler)
	loggedMux := loggingMiddleware(mux)
	http.ListenAndServe(":8080", mux)
}
