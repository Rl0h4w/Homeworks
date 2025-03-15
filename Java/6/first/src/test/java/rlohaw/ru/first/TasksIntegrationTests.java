package rlohaw.ru.first;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TasksIntegrationTests {

	@Autowired
	private TestRestTemplate restTemplate;

	@Test
	@Order(1)
	public void testCreateTask() {
		Task task = new Task();
		task.setTitle("Test Task");
		task.setDescription("Test Description");
		task.setStatus("pending");
		task.setDueDate("2023-12-31");

		ResponseEntity<Task> response = restTemplate.postForEntity("/api/tasks", task, Task.class);
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);

		Task createdTask = response.getBody();
		assertThat(createdTask).isNotNull();
		assertThat(createdTask.getId()).isGreaterThan(0);
		assertThat(createdTask.getTitle()).isEqualTo("Test Task");
	}

	@Test
	@Order(2)
	public void testGetTask() {
		Task task = new Task();
		task.setTitle("Get Task");
		task.setDescription("Some Description");
		task.setStatus("in_progress");
		task.setDueDate("2023-12-31");

		ResponseEntity<Task> createResponse = restTemplate.postForEntity("/api/tasks", task, Task.class);
		int taskId = createResponse.getBody().getId();

		ResponseEntity<Task> getResponse = restTemplate.getForEntity("/api/tasks/" + taskId, Task.class);
		assertThat(getResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
		Task fetchedTask = getResponse.getBody();
		assertThat(fetchedTask).isNotNull();
		assertThat(fetchedTask.getId()).isEqualTo(taskId);
	}

	@Test
	@Order(3)
	public void testUpdateTask() {
		Task task = new Task();
		task.setTitle("Update Task");
		task.setDescription("Old Description");
		task.setStatus("pending");
		task.setDueDate("2023-12-31");

		ResponseEntity<Task> createResponse = restTemplate.postForEntity("/api/tasks", task, Task.class);
		int taskId = createResponse.getBody().getId();

		Task updatedTask = new Task();
		updatedTask.setTitle("Updated Task Title");
		updatedTask.setDescription("New Description");
		updatedTask.setStatus("completed");
		updatedTask.setDueDate("2024-01-01");

		HttpHeaders headers = new HttpHeaders();
		headers.setContentType(MediaType.APPLICATION_JSON);
		HttpEntity<Task> requestEntity = new HttpEntity<>(updatedTask, headers);

		ResponseEntity<Task> updateResponse = restTemplate.exchange(
				"/api/tasks/" + taskId, HttpMethod.PUT, requestEntity, Task.class);
		assertThat(updateResponse.getStatusCode()).isEqualTo(HttpStatus.OK);

		Task resultTask = updateResponse.getBody();
		assertThat(resultTask).isNotNull();
		assertThat(resultTask.getTitle()).isEqualTo("Updated Task Title");
		assertThat(resultTask.getStatus()).isEqualTo("completed");
	}

	@Test
	@Order(4)
	public void testDeleteTask() {
		Task task = new Task();
		task.setTitle("Delete Task");
		task.setDescription("To be deleted");
		task.setStatus("pending");
		task.setDueDate("2023-12-31");

		ResponseEntity<Task> createResponse = restTemplate.postForEntity("/api/tasks", task, Task.class);
		int taskId = createResponse.getBody().getId();

		restTemplate.delete("/api/tasks/" + taskId);

		ResponseEntity<Task> getResponse = restTemplate.getForEntity("/api/tasks/" + taskId, Task.class);
		assertThat(getResponse.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
	}
}
