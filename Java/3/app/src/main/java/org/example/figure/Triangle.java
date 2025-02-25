package org.example.figure;

public class Triangle {
    public Point p1;
    public Point p2;
    public Point p3;

    public Triangle(Point p1, Point p2, Point p3) {
        this.p1 = p1;
        this.p2 = p2;
        this.p3 = p3;
    }

    public double getPerimeter() {
        return p1.getDistanceTo(p2) + p2.getDistanceTo(p3) + p3.getDistanceTo(p1);
    }

}
