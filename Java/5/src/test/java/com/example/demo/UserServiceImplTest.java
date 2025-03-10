package com.example.demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class UserServiceImplTest {

    private UserService userService = new UserServiceImpl();

    @Test
    public void testGetUsername() {
        assertEquals("Spring User", userService.getUsername());
    }
}