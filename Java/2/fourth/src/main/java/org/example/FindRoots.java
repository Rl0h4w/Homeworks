package org.example;


public class FindRoots {
    public static Double[] findRoots(double a, double b, double c) {
        Double dd = b * b - 4 * a * c;
        Double d = Math.sqrt(dd);
        if (dd < 0) {
            // Корней нет
            throw new IllegalArgumentException("Нет корней");
        }
        Double firstRoot = (-b - d) / 2 * a;
        Double sndRoot = (-b + d) / 2 * a;
        return new Double[]{firstRoot, sndRoot};
    }

}