package org.example;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class GetAmountOfLargeSpendsTest {
    @Test
    public void BasicTest() {
        double[] spends = {100, 123, 100, 32};
        double treshold = 1;
        int result = 4;
        Assertions.assertEquals(GetAmountOfLargeSpends.getAmountOfLargeSpends(spends, treshold), result);
    }
}
