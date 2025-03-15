package rlohaw.ru.second.spring_rest.controller;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import java.util.Map;
import org.springframework.web.bind.annotation.RestController;
import rlohaw.ru.second.spring_rest.model.UserDto;

@RestController
@RequestMapping("/user")
public class UserController {

    private final Map<Integer, UserDto> users = new HashMap<>();

    @GetMapping
    public List<UserDto> getAll() {
        return new ArrayList<>(users.values());
    }

    @PostMapping
    public void add(@RequestBody UserDto user) {
        users.put(user.getId(), user);
    }

    @PutMapping("/{id}")
    public void update(@PathVariable Integer id, @RequestBody UserDto user) {
        users.put(id, user);
    }

    @DeleteMapping("/{id}")
    public void delete(@PathVariable Integer id) {
        users.remove(id);
    }

}
