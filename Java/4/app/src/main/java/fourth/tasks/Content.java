package fourth.tasks;

public abstract class Content {
    public Content(String title, String description) {
        this.title = title;
        this.description = description;
    }

    protected String title;
    protected String description;

    public String getTitle() {
        return this.title;
    }

    public String getDescription() {
        return this.description;
    }

    public String setTitle(String title) {
        this.title = title;
        return title;
    }

    public String setDescription(String description) {
        this.description = description;
        return description;
    }
}
