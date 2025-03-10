package main

import (
	"bufio"
	"fmt"
	"os"

	"golang.org/x/crypto/bcrypt"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: passwd <secret_file>")
		os.Exit(1)
	}
	secretFile := os.Args[1]

	fmt.Print("Enter password: ")
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		fmt.Println("Failed to read password")
		os.Exit(1)
	}
	password := scanner.Text()

	hashed, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		fmt.Printf("Error hashing password: %v\n", err)
		os.Exit(1)
	}

	if err = os.WriteFile(secretFile, hashed, 0600); err != nil {
		fmt.Printf("Error writing secret file: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Password saved successfully.")
}
