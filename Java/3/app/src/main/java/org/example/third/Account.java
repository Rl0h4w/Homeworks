package org.example.third;

public class Account {
    private double amountOfMoney;

    public Account(double initialAmount) {
        this.amountOfMoney = initialAmount;
    }

    public double getAmountOfMoney() {
        return this.amountOfMoney;
    }

    public void subtractMoney(double amount) {
        this.amountOfMoney -= amount;
    }

    public void addMoney(double amount) {
        this.amountOfMoney += amount;
    }

    public void transfer(Account other, double amount) {
        if (other.getAmountOfMoney() >= amount) {
            other.subtractMoney(amount);
            this.addMoney(amount);
        } else {
            System.out.println("debt mode?");
        }
    }
}