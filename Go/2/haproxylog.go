package main

import (
	"fmt"
	"regexp"
	"strings"
)

type HaproxyLog struct {
	Date   string
	Params map[string]string
}

func ParseHaproxyLog(log string) (*HaproxyLog, error) {
	r := regexp.MustCompile(`^(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+haproxy\[\d+\]:\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)\s+\[.*?\]\s+(\w+)\s+([\w\/]+)\s+.*$`)
	match := r.FindStringSubmatch(log)
	if match == nil {
		return nil, fmt.Errorf("invalid log format")
	}

	date := match[1]
	clientIP := match[2]
	clientPort := match[3]
	frontend := match[4]
	backend := match[5]

	params := map[string]string{
		"client_port":  clientPort,
		"frontend":     frontend,
		"backend":      backend,
		"process_info": fmt.Sprintf("haproxy[%s]", strings.Split(log, "[")[1][:5]), // Extract PID
		"client_ip":    clientIP,
	}

	return &HaproxyLog{
		Date:   date,
		Params: params,
	}, nil
}

func main() {
	log := "Jan 13 10:23:45 haproxy[12345]: 192.168.1.100:54421 [13/Jan/2025:10:23:45.123] frontend1 backend1/server1 0/0/1/23/24 200 345 - - ---- 1/1/0/0/0 0/0/0/0/0"
	haproxyLog, err := ParseHaproxyLog(log)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Log Details:")
	fmt.Println("severity: INFO")
	for key, value := range haproxyLog.Params {
		fmt.Printf("%s: %s\n", key, value)
	}
}
