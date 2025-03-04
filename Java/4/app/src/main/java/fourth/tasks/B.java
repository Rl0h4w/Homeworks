package fourth.tasks;

public class A {
    A() {
        System.out.println("A");
    }
}

public class B extends A {
    B() {
        System.out.println("B");
    }
}

public class C extends B {
    C() {
        System.out.println("C");
    }

}

class Second {
    public static void main(String[] args) {
        new C();
    }
}