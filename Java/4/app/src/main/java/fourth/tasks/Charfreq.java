package fourth.tasks;

import java.util.HashMap;
import java.util.Map;

public class Charfreq {
    public static Map<Character, Integer> frequencyAnalysis(String input) {
        Map<Character, Integer> frequencyMap = new HashMap<>();

        for (char ch : input.toCharArray()) {
            frequencyMap.put(ch, frequencyMap.getOrDefault(ch, 0) + 1);
        }

        return frequencyMap;
    }

    public static void main(String[] args) {
        String text = "example string";
        Map<Character, Integer> result = frequencyAnalysis(text);

        for (Map.Entry<Character, Integer> entry : result.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}