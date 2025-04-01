package org.example;

import java.sql.ResultSet;
import java.sql.SQLException;

public class Second {

    public void runSecond() {
        Database db = new Database();
        String firstquerry = "INSERT INTO books (title, author, year) VALUES\n" + //
                "('Война и мир', 'Лев Толстой', 1869),\n" + //
                "('Преступление и наказание', 'Фёдор Достоевский', 1866),\n" + //
                "('1984', 'Джордж Оруэлл', 1949),\n" + //
                "('Мастер и Маргарита', 'Михаил Булгаков', NULL),\n" + //
                "('Гарри Поттер и философский камень', 'Дж. К. Роулинг', 1997),\n" + //
                "('Сто лет одиночества', 'Габриэль Гарсия Маркес', 1967),\n" + //
                "('Чужак', 'Альбер Камю', NULL);";
        db.executeUpdate(firstquerry);

        String secondquerry = "DELETE FROM books WHERE year > 1990;";
        db.executeUpdate(secondquerry);

        String thirdquerry = "SELECT title, author\n" + //
                "FROM books\n" + //
                "WHERE year BETWEEN 1801 AND 1900;";

        try {
            ResultSet rs = db.executeQuery(thirdquerry);
            while (rs.next()) {
                System.out.println(rs.getString("title") + " " + rs.getString("author"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        String fourthquerry = "SELECT author\n" + //
                "FROM books\n" + //
                "WHERE author LIKE 'Д%';";

        try {
            ResultSet rs = db.executeQuery(fourthquerry);
            while (rs.next()) {
                System.out.println(rs.getString("author"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        db.closeConnection();
    }
}
