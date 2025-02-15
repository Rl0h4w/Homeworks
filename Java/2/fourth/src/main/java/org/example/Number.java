package org.example;

import org.apache.commons.lang3.math.NumberUtils;

public class Number {
    public static boolean isNumber(String text) {
        return NumberUtils.isCreatable(text);
    }

    public  static  void main(String[] args) {
        System.out.println(isNumber("134"));
        System.out.println(isNumber("14.12"));
        System.out.println(isNumber("2e6"));
        System.out.println(isNumber("0xDEAD"));
    }
}