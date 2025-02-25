package org.example;

import org.example.third.Account;

public class Third {
    public static void main(String[] args) {
        Account acc1 = new Account(100);
        Account acc2 = new Account(150);
        acc1.transfer(acc2, 33);
        System.out.println(acc1.getAmountOfMoney() + ", " + acc2.getAmountOfMoney());
    }
}
