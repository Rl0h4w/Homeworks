package org.example;

public class GetAmountOfLargeSpends {
    public static int getAmountOfLargeSpends(double[] spends, double treshold) {
        int count = 0;
        for (double amount : spends) {
            if (amount > treshold) {
                count++;
            }
        }
        return count;
    }

}
