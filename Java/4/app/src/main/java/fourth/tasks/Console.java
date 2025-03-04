package fourth.tasks;

import java.util.ArrayList;
import java.util.Scanner;

public class Console {
    public static void main(String[] args) {
        ArrayList<String> list = new ArrayList<>();
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("Меню:");
            System.out.println("1. Добавить элемент");
            System.out.println("2. Показать элементы");
            System.out.println("3. Выход");
            System.out.print("Выберите действие: ");

            int choice = scanner.nextInt();
            scanner.nextLine();

            switch (choice) {
                case 1:
                    System.out.print("Введите строку: ");
                    String item = scanner.nextLine();
                    list.add(item);
                    System.out.println("Элемент добавлен!\n");
                    break;
                case 2:
                    System.out.println("Список элементов:");
                    for (String s : list) {
                        System.out.println("- " + s);
                    }
                    System.out.println();
                    break;
                case 3:
                    System.out.println("Выход...");
                    scanner.close();
                    return;
                default:
                    System.out.println("Неверный выбор, попробуйте снова.\n");
            }
        }
    }
}
