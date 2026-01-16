package com.micrograd.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 資料載入器
 * 讀取 names.txt 並提取 bigram 資料
 */
public class DataLoader {

    private final List<String> words;
    private final Vocabulary vocab;

    public DataLoader(String filepath, Vocabulary vocab) throws IOException {
        this.vocab = vocab;
        this.words = loadWords(filepath);
    }

    private List<String> loadWords(String filepath) throws IOException {
        List<String> result = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim().toLowerCase();
                if (!line.isEmpty() && isValidWord(line)) {
                    result.add(line);
                }
            }
        }

        return result;
    }

    /**
     * 檢查單字是否只包含有效字元（a-z）
     */
    private boolean isValidWord(String word) {
        for (char c : word.toCharArray()) {
            if (!vocab.contains(c) && c != Vocabulary.SPECIAL_TOKEN) {
                return false;
            }
        }
        return true;
    }

    /**
     * 取得所有單字
     */
    public List<String> getWords() {
        return words;
    }

    /**
     * 取得單字數量
     */
    public int getWordCount() {
        return words.size();
    }

    /**
     * 提取所有 bigram（輸入-目標對）
     * 回傳 int[2][]，其中 [0] 是輸入索引，[1] 是目標索引
     */
    public int[][] getBigrams() {
        List<Integer> inputs = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();

        for (String word : words) {
            // 加上開始和結束符號
            String padded = Vocabulary.SPECIAL_TOKEN + word + Vocabulary.SPECIAL_TOKEN;

            for (int i = 0; i < padded.length() - 1; i++) {
                char ch1 = padded.charAt(i);
                char ch2 = padded.charAt(i + 1);
                inputs.add(vocab.encode(ch1));
                targets.add(vocab.encode(ch2));
            }
        }

        int[][] result = new int[2][inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            result[0][i] = inputs.get(i);
            result[1][i] = targets.get(i);
        }

        return result;
    }

    /**
     * 取得 bigram 數量
     */
    public int getBigramCount() {
        int count = 0;
        for (String word : words) {
            count += word.length() + 1;  // 每個字有 len+1 個 bigram
        }
        return count;
    }

    /**
     * 印出一些統計資訊
     */
    public void printStats() {
        System.out.println("=== 資料統計 ===");
        System.out.println("單字數量: " + words.size());
        System.out.println("Bigram 數量: " + getBigramCount());

        // 找最短和最長的單字
        int minLen = Integer.MAX_VALUE;
        int maxLen = 0;
        for (String word : words) {
            minLen = Math.min(minLen, word.length());
            maxLen = Math.max(maxLen, word.length());
        }
        System.out.println("最短單字長度: " + minLen);
        System.out.println("最長單字長度: " + maxLen);

        // 印出前幾個單字
        System.out.println("前 10 個單字: ");
        for (int i = 0; i < Math.min(10, words.size()); i++) {
            System.out.println("  " + words.get(i));
        }
    }

    /**
     * 印出某個單字的 bigram
     */
    public void printBigrams(String word) {
        System.out.println("單字 '" + word + "' 的 bigrams:");
        String padded = Vocabulary.SPECIAL_TOKEN + word + Vocabulary.SPECIAL_TOKEN;

        for (int i = 0; i < padded.length() - 1; i++) {
            char ch1 = padded.charAt(i);
            char ch2 = padded.charAt(i + 1);
            int idx1 = vocab.encode(ch1);
            int idx2 = vocab.encode(ch2);
            System.out.printf("  '%c' -> '%c'  (%d -> %d)\n", ch1, ch2, idx1, idx2);
        }
    }
}