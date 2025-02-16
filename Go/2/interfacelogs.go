package main

import (
	"fmt"
	"regexp"
	"time"
)

type AppLog struct {
	AppID    string
	Severity string
	Action   string
	Date     time.Time
}

func (log AppLog) String() string {
	return fmt.Sprintf("AppLog - AppID: %s, Severity: %s, Action: %s", log.AppID, log.Severity, log.Action)
}

func (log *AppLog) GetSeverity() string {
	return log.Severity
}

func ParseLog(line string) (*AppLog, error) {
	pattern := regexp.MustCompile(`^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(DEBUG|INFO|WARNING|ERROR)\s+\[([^\]]+)\]\s+(.+)$`)
	matches := pattern.FindStringSubmatch(line)
	if matches == nil || len(matches) < 5 {
		return nil, fmt.Errorf("неверный формат лога")
	}

	date, err := time.Parse("2006-01-02 15:04:05", matches[1])
	if err != nil {
		return nil, err
	}

	return &AppLog{
		Date:     date,
		Severity: matches[2],
		AppID:    matches[3],
		Action:   matches[4],
	}, nil
}

func main() {
	input := "2025-01-13 12:45:33 DEBUG [App5] Cache miss for 'user_123' on request to endpoint /user/profile"

	appLog, err := ParseLog(input)
	if err != nil {
		fmt.Println("Ошибка парсинга лога:", err)
		return
	}

	fmt.Println(appLog)
	fmt.Println("Severity:", appLog.GetSeverity())
}
