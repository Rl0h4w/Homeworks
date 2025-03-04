package fourth.tasks;

public class Song extends Content {
    private String band;
    private int year;
    private String album;

    Song(String title, String description, String band, int year, String album) {
        super(title, description);
        this.band = band;
        this.year = year;
        this.album = album;
    }

    public String getBand() {
        return band;
    }

    public String getAlbum() {
        return album;
    }

    public int getYear() {
        return year;
    }

}
