package org.example;

import org.example.figure.Point;
import org.example.figure.Triangle;

public class First {

    public static void main(String[] args) {
        Point p1 = new Point(0., 1.);
        Point p2 = new Point(1., 0);
        Point p3 = new Point(0., 0);
        Triangle trig = new Triangle(p1, p2, p3);
        System.out.println(trig.getPerimeter());
    }
}
