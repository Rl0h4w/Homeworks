package com.example.demo;

import org.springframework.stereotype.Service;

@Service
public class UserServiceImpl implements UserService {
    @Override
    public String getUsername() {
        return "Spring User";
    }
}
