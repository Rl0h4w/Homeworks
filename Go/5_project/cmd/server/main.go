package main

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"golang.org/x/crypto/bcrypt"
)

const (
	authTimeout = 15 * time.Second
	prompt      = "> "
)

func handleConnection(conn net.Conn, passwordHash []byte) {
	defer conn.Close()

	conn.Write([]byte("Password: "))
	conn.SetDeadline(time.Now().Add(authTimeout))
	reader := bufio.NewReader(conn)
	input, err := reader.ReadString('\n')
	if err != nil {
		return
	}
	password := strings.TrimSpace(input)
	if err = bcrypt.CompareHashAndPassword(passwordHash, []byte(password)); err != nil {
		conn.Write([]byte("Invalid password\n"))
		return
	}
	conn.SetDeadline(time.Time{})
	conn.Write([]byte("Welcome!\n" + prompt))

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return
		}
		cmdLine := strings.TrimSpace(line)
		if cmdLine == "" || strings.HasPrefix(cmdLine, "#") {
			conn.Write([]byte(prompt))
			continue
		}
		if cmdLine == "exit" {
			return
		}
		args := strings.Fields(cmdLine)
		if len(args) == 0 {
			conn.Write([]byte(prompt))
			continue
		}
		cmd := exec.CommandContext(context.Background(), args[0], args[1:]...)
		output, err := cmd.CombinedOutput()
		if err != nil {
			conn.Write([]byte(fmt.Sprintf("Error: %v\n", err)))
		}
		conn.Write(output)
		conn.Write([]byte(prompt))
	}
}

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: server <secret_file> <address>")
		os.Exit(1)
	}
	secretFile := os.Args[1]
	address := os.Args[2]

	passwordHash, err := os.ReadFile(secretFile)
	if err != nil {
		fmt.Printf("Error reading secret file: %v\n", err)
		os.Exit(1)
	}

	ln, err := net.Listen("tcp", address)
	if err != nil {
		fmt.Printf("Error listening on %s: %v\n", address, err)
		os.Exit(1)
	}
	defer ln.Close()
	fmt.Printf("Server listening on %s\n", address)

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigs
		fmt.Println("\nShutting down server...")
		ln.Close()
		os.Exit(0)
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			break
		}
		go handleConnection(conn, passwordHash)
	}
}
