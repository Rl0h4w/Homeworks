package rlohaw.ru.second.spring_rest.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@ToString
public class BookDto {
    private Long id;
    private String name;
    private String author;
}
