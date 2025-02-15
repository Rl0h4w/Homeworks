package org.example; // Такой же пакет, как и у Main-класса

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CartTest {

    @Test // Аннотация, которая говорит о том, что этот метод должен запускаться как тест
    public void testApplyDiscountBaseCase() {
        // Arrange
        double[] cart = {100, 200, 300};
        double discount = 20;
        double[] expected = {80.0, 160.0, 240.0};

        // Act
        double[] result = Cart.applyDiscount(cart, discount);

        // Assert
        // Первый аргумент — ожидаемый результат, второй — реальный
        assertEquals((int)expected[0], (int)result[0]);
        assertEquals((int)expected[1], (int)result[1]);
        assertEquals((int)expected[2], (int)result[2]);
    }
    @Test // Аннотация, которая говорит о том, что этот метод должен запускаться как тест
    public void testApplyDiscountFirstEdgeCase() {
        // Arrange
        double[] cart = {100, 200, 300};
        double discount = 0;
        double[] expected = {100, 200, 300};

        // Act
        double[] result = Cart.applyDiscount(cart, discount);

        // Assert
        // Первый аргумент — ожидаемый результат, второй — реальный
        assertEquals((int)expected[0], (int)result[0]);
        assertEquals((int)expected[1], (int)result[1]);
        assertEquals((int)expected[2], (int)result[2]);
    }
    @Test // Аннотация, которая говорит о том, что этот метод должен запускаться как тест
    public void testApplyDiscountSecondEdgeCase() {
        // Arrange
        double[] cart = {100, 200, 300};
        double discount = 0;
        double[] expected = {100, 200, 300};

        // Act
        double[] result = Cart.applyDiscount(cart, discount);

        // Assert
        // Первый аргумент — ожидаемый результат, второй — реальный
        assertEquals((int)expected[0], (int)result[0]);
        assertEquals((int)expected[1], (int)result[1]);
        assertEquals((int)expected[2], (int)result[2]);
    }
    @Test // Исправленный тест для пустой корзины
    public void testEmptyCart() {
        double[] cart = {};
        double discount = 10;
        double[] expected = {};

        double[] result = Cart.applyDiscount(cart, discount);

        assertArrayEquals(expected, result);
    }
    @Test
    public void testSingleItemCart() {
        double[] cart = {150};
        double discount = 25;
        double[] expected = {112.5};

        double[] result = Cart.applyDiscount(cart, discount);

        assertArrayEquals(expected, result);
    }


}
