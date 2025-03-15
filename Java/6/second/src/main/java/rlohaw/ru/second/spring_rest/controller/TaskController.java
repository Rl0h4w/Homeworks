package rlohaw.ru.second.spring_rest.controller;

import rlohaw.ru.second.spring_rest.model.TaskDto;
import rlohaw.ru.second.spring_rest.service.TaskService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "Task API", description = "Управление задачами")
@RestController
@RequestMapping("/tasks")
@RequiredArgsConstructor
public class TaskController {

    private final TaskService taskService;

    @Operation(summary = "Получить все задачи")
    @GetMapping
    public List<TaskDto> getAllTasks() {
        return taskService.getAllTasks();
    }

    @Operation(summary = "Добавить новую задачу")
    @PostMapping
    public void addTask(@Valid @RequestBody TaskDto task) {
        taskService.addTask(task);
    }

    @Operation(summary = "Обновить задачу по ID")
    @PutMapping("/{id}")
    public void updateTask(
            @PathVariable Long id,
            @Valid @RequestBody TaskDto task) {
        taskService.updateTask(id, task);
    }

    @Operation(summary = "Удалить задачу по ID")
    @DeleteMapping("/{id}")
    public void deleteTask(@PathVariable Long id) {
        taskService.deleteTask(id);
    }
}
