import java.util.Scanner;


public class ScalarProduct {
    public static int[] vectorProduct(int[] vector_a, int[] vector_b){
    int size_a = vector_a.length;
    int size_b = vector_b.length;
        if (size_a==size_b) {
        int[] vector_c = new int[size_a];
        for (int i = 0; i < size_a; ++i) {
            vector_c[i] = vector_a[i] * vector_b[i];
        }
        return vector_c;
    }
    return null;
    }
    
    public static int[] scanVector(Scanner scanner){
        int size_a = 0;
        size_a = scanner.nextInt();
        int[] vector_a = new int[size_a];
        for (int i = 0; i < size_a; ++i) {
            vector_a[i] = scanner.nextInt();
        }
        return vector_a;
    }
    public static void printVector(int[] vector) {
    if (vector != null) {
            for (int num: vector) {
                System.out.print(Integer.toString(num)+" ");
            }
        } else {
            System.out.print("Different Dimensions");
        }
    }
    public static void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        int[] vector_a = scanVector(scanner);
        int[] vector_b = scanVector(scanner);
        scanner.close();
        int[] res = vectorProduct(vector_a, vector_b);
        printVector(res);
    }
}