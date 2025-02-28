package app.src.main.java.fourth.tasks;

public abstract class Stats {
    private String title;
    private int year;
    private String city;

    Stats(String title, int year, String city) {
        this.title = title;
        this.year = year;
        this.city = city;
    }

    public String getTitle() {
        return title;
    }

    public int getYear() {
        return year;
    }

    public String getCity() {
        return city;
    }
}

public class Book extends Stats {
    private String author;
    private String describe;
    private String genre;

    public Book(String title, int year, String city, String author, String describe, String genre) {
        super(title, year, city);
        this.author = author;
        this.describe = describe;
        this.genre = genre;
    }

    public String getAuthor() {
        return author;
    }

    public String getDescribe() {
        return describe;
    }

    public String getGenre() {
        return genre;

    }
}

public class Newspaper extends Stats {
    private int day;
    private int month;
    private String[] titles;

    Newspaper(String title, int year, String city, int day, int month, String[] titles) {
        super(title, year, city);
        this.day = day;
        this.month = month;
        this.titles = titles;
    }

    public int getDay() {
        return day;
    }

    public int getMonth() {
        return month;
    }

    public String[] getTitles() {
        return titles;
    }
}

public class researchPaper extends Stats {
    private String author;
    private String coauthor;
    private String reviewer;
    private String field;

    researchPaper(String title, int year, String city, String author, String coauthor, String reviewer, String field) {
        super(title, year, city);
        this.author = author;
        this.coauthor = coauthor;
        this.reviewer = reviewer;
        this.field = field;
    }
}
