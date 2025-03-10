package com.example.demo;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class GreetingController {

    private final UserService userService;

    public GreetingController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/greet")
    public String greet(Model model) {
        model.addAttribute("username", userService.getUsername());
        return "greeting";
    }
}