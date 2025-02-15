package org.example;

public class Cart {


    public static double[] applyDiscount(double[] cart, double discount) {
        if (discount < 0 || discount > 100) {
            throw new IllegalArgumentException("Размер скидки должен быть в диапазоне от 0 до 100.");
        }

        double discountFactor = 1 - discount / 100.0;
        double[] discountedCart = new double[cart.length];

        for (int i = 0; i < cart.length; i++) {
            discountedCart[i] = cart[i] * discountFactor;
        }

        return discountedCart;
    }
}
