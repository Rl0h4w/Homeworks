package rlohaw.ru.second.spring_rest.service;

import rlohaw.ru.second.spring_rest.exception.HttpStatusException;
import rlohaw.ru.second.spring_rest.model.TaskDto;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class TaskService {
    private final Map<Long, TaskDto> tasks = new HashMap<>();

    public List<TaskDto> getAllTasks() {
        return List.copyOf(tasks.values());
    }

    public void addTask(TaskDto task) {
        if (tasks.containsKey(task.getId())) {
            throw new HttpStatusException(HttpStatus.CONFLICT, "Задача с ID " + task.getId() + " уже существует");
        }
        tasks.put(task.getId(), task);
    }

    public void updateTask(Long id, TaskDto task) {
        if (!tasks.containsKey(id)) {
            throw new HttpStatusException(HttpStatus.NOT_FOUND, "Задача с ID " + id + " не найдена");
        }
        task.setId(id);
        tasks.put(id, task);
    }

    public void deleteTask(Long id) {
        if (!tasks.containsKey(id)) {
            throw new HttpStatusException(HttpStatus.NOT_FOUND, "Задача с ID " + id + " не найдена");
        }
        tasks.remove(id);
    }
}
