package org.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class IsValidDataTest {
    @Test
    public void BaseTest() {
        double[] arr = {123.2, 123.1};
        Assertions.assertEquals(IsValidData.isValidData(arr, 123.12, 123.19), false);
        Assertions.assertEquals(IsValidData.isValidData(arr, 123.1, 123.2), true);
    }
}
