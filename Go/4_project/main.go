package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"
)

type Task struct {
	Time    time.Duration
	Command string
	Args    []string
}

var pattern = regexp.MustCompile(`^(\d{2}):(\d{2}):(\d{2})\s+(.+)$`)

func parse(filename string) ([]Task, error) {
	inputFile, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer inputFile.Close()

	var taskList []Task
	lineScanner := bufio.NewScanner(inputFile)
	for lineScanner.Scan() {
		trimmedLine := strings.TrimSpace(lineScanner.Text())
		if trimmedLine == "" || strings.HasPrefix(trimmedLine, "#") || isEmo(trimmedLine) {
			continue
		}

		timeMatch := pattern.FindStringSubmatch(trimmedLine)
		if timeMatch == nil {
			fmt.Println("Invalid input", trimmedLine)
			continue
		}

		hours, _ := strconv.Atoi(timeMatch[1])
		minutes, _ := strconv.Atoi(timeMatch[2])
		seconds, _ := strconv.Atoi(timeMatch[3])
		cmdParts := strings.Fields(timeMatch[4])
		if len(cmdParts) == 0 {
			continue
		}

		taskDuration := time.Duration(hours)*time.Hour +
			time.Duration(minutes)*time.Minute +
			time.Duration(seconds)*time.Second

		taskList = append(taskList, Task{
			Time:    taskDuration,
			Command: cmdParts[0],
			Args:    cmdParts[1:],
		})
	}

	sort.Slice(taskList, func(i, j int) bool {
		return taskList[i].Time < taskList[j].Time
	})

	return taskList, lineScanner.Err()
}

func isEmo(inputLine string) bool {
	if inputLine == "" {
		return false
	}
	firstRune, _ := utf8.DecodeRuneInString(inputLine)
	return unicode.Is(unicode.So, firstRune)
}

func run(currentTask Task, timeout time.Duration) {
	parentCtx := context.Background()
	taskCtx, cancelFunc := context.WithTimeout(parentCtx, timeout)
	defer cancelFunc()

	cmdInstance := exec.CommandContext(taskCtx, currentTask.Command, currentTask.Args...)
	cmdInstance.Stdout = os.Stdout
	cmdInstance.Stderr = os.Stderr
	fmt.Println("Executing task", currentTask.Command)

	if err := cmdInstance.Run(); err != nil {
		fmt.Println("Task error:", err)
	}
}

func plan(taskList []Task) {
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	nextTimes := make([]time.Duration, len(taskList))
	for i := range taskList {
		if i < len(taskList)-1 {
			nextTimes[i] = taskList[i+1].Time
		} else {
			nextTimes[i] = taskList[0].Time + 24*time.Hour
		}
	}

	for {
		currentTime := time.Now()
		midnight := currentTime.Truncate(24 * time.Hour)
		currentDuration := currentTime.Sub(midnight)

		for idx, currentTask := range taskList {
			delayDuration := currentTask.Time - currentDuration
			if delayDuration < 0 {
				delayDuration += 24 * time.Hour
			}

			select {
			case <-time.After(delayDuration):
				run(currentTask, nextTimes[idx]-currentTask.Time)
			case <-signalChan:
				fmt.Println("Terminating")
				return
			}
		}
	}
}

func main() {
	taskList, err := parse(os.Args[1])
	if err != nil {
		os.Exit(1)
	}
	plan(taskList)
}
