package org.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class FindRootsTest {
    @Test
    public void TestComplexDiscriminant() {
        int b = 1;
        int a = 2;
        int c = 2;
        Assertions.assertThrows(IllegalArgumentException.class, () -> FindRoots.findRoots(a, b, c));
    }
}
