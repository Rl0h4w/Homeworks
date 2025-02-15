package org.example;

public class IsValidData {
    public static boolean isValidData(double[] data, double min, double max) {
        for (double entry : data) {
            if (!(entry >= min && entry <= max)){
                return  false;
            }
        }
        return true;
    }
}
