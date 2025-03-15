package rlohaw.ru.first;

import jakarta.validation.constraints.Pattern;

public class Task {
    private int id;
    private String title;
    private String description;

    @Pattern(regexp = "^(pending|in_progress|completed)$", message = "Status must be one of: pending, in_progress, completed")
    private String status;

    @Pattern(regexp = "^\\d{4}-\\d{2}-\\d{2}$", message = "Due date must be in format YYYY-MM-DD")
    private String dueDate;

    public int getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }

    public String getStatus() {
        return status;
    }

    public String getDueDate() {
        return dueDate;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setDueDate(String dueDate) {
        this.dueDate = dueDate;
    }

    @Override
    public String toString() {
        return "Task{" +
                "id=" + id +
                ", title='" + title + '\'' +
                ", description='" + description + '\'' +
                ", status='" + status + '\'' +
                ", dueDate='" + dueDate + '\'' +
                '}';
    }
}
