package org.example;

public class First {
    public void runFirst() {
        Database db = new Database();
        String createTableQuery = "CREATE TABLE IF NOT EXISTS books (id SERIAL PRIMARY KEY, title text not null, author text not null, year int);";
        db.executeUpdate(createTableQuery);

        String alterTableAddColumnQuery = "ALTER TABLE books ADD COLUMN IF NOT EXISTS genre text;";
        db.executeUpdate(alterTableAddColumnQuery);

        String alterTableDropColumnQuery = "ALTER TABLE books DROP COLUMN IF EXISTS publishing_house;";
        db.executeUpdate(alterTableDropColumnQuery);

        db.closeConnection();
    }
}
