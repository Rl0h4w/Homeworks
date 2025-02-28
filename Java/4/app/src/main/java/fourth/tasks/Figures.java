package app.src.main.java.fourth.tasks;

public abstract class Figures {
    public String color;

    Figures(String color) {
        this.color = color;
    }

    protected abstract double getPerimeter();

    protected abstract double getArea();

    void printInfo() {
        System.out.println("Цвет фигуры: " + color);
        System.out.println("Площадь фигуры" + getArea());
        System.out.println("Периметр фигуры" + getPerimeter());
    }
}

public class Triangle extends Figures {
    private double a;
    private double b;
    private double c;

    public Triangle(String color, double a, double b, double c) {
        super(color);
        this.a = a;
        this.b = b;
        this.c = c;
    }

    @Override
    protected double getPerimeter() {
        return a + b + c;
    }

    @Override
    protected double getArea() {
        double p = getPerimeter() / 2;
        return Math.sqrt(p * (p - a) * (p - b) * (p - c));
    }

}

public class Circle extends Figures {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    @Override
    protected double getPerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    protected double getArea() {
        return Math.PI * radius * radius;
    }
}

public class Rectangle extends Figures {
    private double a;
    private double b;

    public Rectangle(String color, double a, double b) {
        super(color);
        this.a = a;
        this.b = b;
    }

    @Override
    protected double getPerimeter() {
        return 2 * (a + b);
    }

    @Override
    protected double getArea() {
        return a * b;
    }
}
