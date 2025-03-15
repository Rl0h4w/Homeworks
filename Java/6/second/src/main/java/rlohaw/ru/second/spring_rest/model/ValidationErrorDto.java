package rlohaw.ru.second.spring_rest.model;

import lombok.Getter;
import lombok.Setter;
import java.util.List;

@Getter
@Setter
public class ValidationErrorDto {
    private String message;
    private List<ValidationErrorDetail> details;

    public ValidationErrorDto(String message, List<ValidationErrorDetail> details) {
        this.message = message;
        this.details = details;
    }
}
