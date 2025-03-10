package com.example.demo;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
public class MessageController {

    @GetMapping("/message")
    public String showForm(Model model) {
        model.addAttribute("message", new Message());
        return "message-form";
    }

    @PostMapping("/message")
    public String submitForm(@ModelAttribute Message message, Model model) {
        model.addAttribute("message", message);
        return "result";
    }
}