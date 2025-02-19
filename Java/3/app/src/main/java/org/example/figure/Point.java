package org.example.figure;

import java.lang.Math;

public class Point {
    double x;
    double y;

    public double get_distance(Point a, Point b) {
        return Math.sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
    }

    
}
