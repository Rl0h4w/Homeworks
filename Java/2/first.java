import java.util.Scanner;

public class first {

    public static void validateInt(int value, int min, int max) {
        if (value < min || value > max) {
            throw new IllegalArgumentException("Значение должно быть между " + min + " и " + max + " включительно, но не " + value + ".");
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Введите целое число от 1 до 10 включительно: ");

        if (scanner.hasNextInt()) {
            int number = scanner.nextInt();

            try {
                validateInt(number, 1, 10);
                System.out.println("Число валидно.");
            } catch (IllegalArgumentException e) {
                System.out.println("Ошибка: " + e.getMessage());
            }
        } else {
            System.out.println("Ошибка: Введено некорректное значение. Пожалуйста, введите целое число.");
        }

        scanner.close();
    }
}