package fourth.tasks;

public class Photo extends Content {
    private int height;
    private int weight;

    public Photo(String title, String description, int height, int weight) {
        super(title, description);
        this.height = height;
        this.weight = weight;
    }

    public int getHeight() {
        return height;
    }

    public int getWeight() {
        return weight;

    }

    public int setWeight(int weight) {
        this.weight = weight;
        return weight;
    }

    public int setHeight(int height) {
        this.height = height;
        return height;
    }

}