package fourth.tasks;

public class Film extends Content {
    public Film(String title, String description, String director, String[] actors) {
        super(title, description);
        this.director = director;
        this.actors = actors;
    }

    private String director;
    private String[] actors;

    public String[] getActors() {
        return actors;
    }

    public String getDirector() {
        return director;
    }

}