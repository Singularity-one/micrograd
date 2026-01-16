package com.micrograd.data;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Vocabulary 詞彙表測試")
class VocabularyTest {

    private Vocabulary vocab;

    @BeforeEach
    void setUp() {
        vocab = new Vocabulary();
    }

    @Test
    @DisplayName("詞彙表大小應為 27")
    void testVocabSize() {
        assertEquals(27, vocab.size());
    }

    @Test
    @DisplayName("特殊符號 '.' 的索引應為 0")
    void testSpecialToken() {
        assertEquals(0, vocab.encode('.'));
        assertEquals('.', vocab.decode(0));
    }

    @Test
    @DisplayName("字母 'a' 的索引應為 1")
    void testLetterA() {
        assertEquals(1, vocab.encode('a'));
        assertEquals('a', vocab.decode(1));
    }

    @Test
    @DisplayName("字母 'z' 的索引應為 26")
    void testLetterZ() {
        assertEquals(26, vocab.encode('z'));
        assertEquals('z', vocab.decode(26));
    }

    @Test
    @DisplayName("所有字母編碼解碼應一致")
    void testAllLetters() {
        for (char c = 'a'; c <= 'z'; c++) {
            int idx = vocab.encode(c);
            char decoded = vocab.decode(idx);
            assertEquals(c, decoded);
        }
    }

    @Test
    @DisplayName("編碼未知字元應拋出例外")
    void testUnknownChar() {
        assertThrows(IllegalArgumentException.class, () -> {
            vocab.encode('1');
        });
        assertThrows(IllegalArgumentException.class, () -> {
            vocab.encode('A');
        });
    }

    @Test
    @DisplayName("解碼無效索引應拋出例外")
    void testInvalidIndex() {
        assertThrows(IllegalArgumentException.class, () -> {
            vocab.decode(-1);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            vocab.decode(27);
        });
    }

    @Test
    @DisplayName("contains 方法應正確判斷")
    void testContains() {
        assertTrue(vocab.contains('.'));
        assertTrue(vocab.contains('a'));
        assertTrue(vocab.contains('z'));
        assertFalse(vocab.contains('A'));
        assertFalse(vocab.contains('1'));
    }
}
