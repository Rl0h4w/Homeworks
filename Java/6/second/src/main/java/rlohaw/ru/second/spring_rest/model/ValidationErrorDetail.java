package rlohaw.ru.second.spring_rest.model;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ValidationErrorDetail {
    private String field;
    private String error;

    public ValidationErrorDetail(String field, String error) {
        this.field = field;
        this.error = error;
    }
}