import java.util.Scanner;

public class second {

    public static void validateInt(int value, int min, int max) {
        if (value < min || value > max) {
            throw new IllegalArgumentException("Значение должно быть между " + min + " и " + max + " включительно, но не " + value + ".");
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        try (scanner) {
            System.out.print("Введите строку, представляющую целое число от 1 до 10 включительно: ");
            String inputString = scanner.nextLine();
            int number = Integer.parseInt(inputString);
            validateInt(number, 1, 10);
            System.out.println("Число корректное.");
        } catch (NumberFormatException e) {
            System.out.println("Строка не является числом.");
        } catch (IllegalArgumentException e) {
            System.out.println("Ошибка: " + e.getMessage());
        }
    }
}