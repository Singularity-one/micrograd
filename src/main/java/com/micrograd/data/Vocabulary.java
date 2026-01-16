package com.micrograd.data;

import java.util.HashMap;
import java.util.Map;

/**
 * 字元詞彙表
 * 管理字元與索引之間的對應關係
 *
 * 包含 27 個字元：
 * - '.' (索引 0): 特殊符號，代表開始和結束
 * - 'a'-'z' (索引 1-26): 26 個英文字母
 */
public class Vocabulary {

    public static final char SPECIAL_TOKEN = '.';
    public static final int VOCAB_SIZE = 27;

    private final Map<Character, Integer> stoi;  // string to index
    private final Map<Integer, Character> itos;  // index to string

    public Vocabulary() {
        this.stoi = new HashMap<>();
        this.itos = new HashMap<>();
        buildVocabulary();
    }

    private void buildVocabulary() {
        // 特殊符號 '.' 在索引 0
        stoi.put(SPECIAL_TOKEN, 0);
        itos.put(0, SPECIAL_TOKEN);

        // 'a' 到 'z' 在索引 1-26
        for (int i = 0; i < 26; i++) {
            char c = (char) ('a' + i);
            int index = i + 1;
            stoi.put(c, index);
            itos.put(index, c);
        }
    }

    /**
     * 字元轉索引
     */
    public int encode(char c) {
        Integer index = stoi.get(c);
        if (index == null) {
            throw new IllegalArgumentException("Unknown character: " + c);
        }
        return index;
    }

    /**
     * 索引轉字元
     */
    public char decode(int index) {
        Character c = itos.get(index);
        if (c == null) {
            throw new IllegalArgumentException("Unknown index: " + index);
        }
        return c;
    }

    /**
     * 檢查字元是否在詞彙表中
     */
    public boolean contains(char c) {
        return stoi.containsKey(c);
    }

    /**
     * 取得詞彙表大小
     */
    public int size() {
        return VOCAB_SIZE;
    }

    /**
     * 取得特殊符號的索引
     */
    public int getSpecialTokenIndex() {
        return stoi.get(SPECIAL_TOKEN);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Vocabulary (size=").append(VOCAB_SIZE).append("):\n");
        for (int i = 0; i < VOCAB_SIZE; i++) {
            char c = itos.get(i);
            String display = (c == SPECIAL_TOKEN) ? "." : String.valueOf(c);
            sb.append(String.format("  %2d -> '%s'\n", i, display));
        }
        return sb.toString();
    }
}