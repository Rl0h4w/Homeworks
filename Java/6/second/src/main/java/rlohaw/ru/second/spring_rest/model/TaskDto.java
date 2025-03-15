package rlohaw.ru.second.spring_rest.model;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class TaskDto {
    @NotNull(message = "ID не может быть пустым")
    private Long id;

    @NotEmpty(message = "Название задачи не может быть пустым")
    private String name;

    @NotEmpty(message = "Автор не может быть пустым")
    private String author;

    private String description;

    @NotNull(message = "Приоритет не может быть пустым")
    private TaskPriority priority;
}