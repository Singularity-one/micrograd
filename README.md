# Java Micrograd

A scalar-valued autograd engine and neural network library in Java.  
Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## 簡介

這是 Karpathy 的 [micrograd](https://github.com/karpathy/micrograd) 的 Java 實現版本，用於學習：

- 自動微分（Automatic Differentiation）
- 反向傳播（Backpropagation）
- 神經網路基礎（Neural Networks）

## 專案結構
```
java-micrograd/
├── pom.xml
├── README.md
├── output/                         # 生成的圖片
└── src/
    ├── main/java/com/micrograd/
    │   ├── engine/
    │   │   └── Value.java          # 自動微分核心
    │   ├── nn/
    │   │   ├── Module.java         # 神經網路介面
    │   │   ├── Neuron.java         # 神經元
    │   │   ├── Layer.java          # 層
    │   │   └── MLP.java            # 多層感知器
    │   ├── viz/
    │   │   └── GraphVisualizer.java # 計算圖視覺化
    │   └── Main.java               # 訓練範例
    └── test/java/com/micrograd/
        ├── engine/
        │   └── ValueTest.java      # Value 測試
        └── nn/
            └── MLPTest.java        # MLP 測試
```

## 環境需求

- **Java 17** 或以上
- **Maven 3.6** 或以上
- **Graphviz**（用於計算圖視覺化，可選）

## 安裝 Graphviz（可選）

計算圖視覺化功能需要安裝 Graphviz：

### Windows
```powershell
winget install graphviz
```

安裝後需要將 `C:\Program Files\Graphviz\bin` 加入 PATH 環境變數。

### macOS
```bash
brew install graphviz
```

### Ubuntu/Debian
```bash
sudo apt install graphviz
```

### 驗證安裝
```bash
dot -version
```

## 快速開始

### 編譯
```bash
mvn compile
```

### 執行訓練範例
```bash
mvn exec:java -Dexec.mainClass="com.micrograd.Main"
```

### 執行測試
```bash
mvn test
```

### 打包
```bash
mvn package
java -jar target/java-micrograd-1.0.0.jar
```

## 使用範例

### 基本運算與自動微分
```java
import com.micrograd.engine.Value;

// 建立計算圖
Value a = new Value(2.0);
Value b = new Value(-3.0);
Value c = new Value(10.0);

Value d = a.mul(b).add(c);  // d = a * b + c = -6 + 10 = 4
d.backward();               // 反向傳播

System.out.println("d = " + d.getData());      // 4.0
System.out.println("da/dd = " + a.getGrad());  // -3.0 (= b)
System.out.println("db/dd = " + b.getGrad());  // 2.0  (= a)
System.out.println("dc/dd = " + c.getGrad());  // 1.0
```

### 建立神經網路
```java
import com.micrograd.nn.MLP;
import com.micrograd.engine.Value;
import java.util.*;

// 建立 MLP: 3 輸入 → 4 隱藏 → 4 隱藏 → 1 輸出
MLP model = new MLP(Arrays.asList(3, 4, 4, 1));

// 前向傳播
List<Value> input = Arrays.asList(
    new Value(2.0),
    new Value(3.0),
    new Value(-1.0)
);
Value output = model.forwardSingle(input);

// 反向傳播
output.backward();

// 查看參數數量
System.out.println("參數數量: " + model.numParameters());
```

### 訓練神經網路
```java
// 資料
double[][] xs = {{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
double[] ys = {1.0, -1.0, -1.0, 1.0};

// 訓練循環
for (int epoch = 0; epoch < 100; epoch++) {
    // 前向傳播 + 計算損失
    Value loss = new Value(0);
    for (int i = 0; i < xs.length; i++) {
        List<Value> input = toValueList(xs[i]);
        Value pred = model.forwardSingle(input);
        loss = loss.add(pred.sub(ys[i]).pow(2));
    }
    
    // 反向傳播
    model.zeroGrad();
    loss.backward();
    
    // 梯度下降更新
    for (Value p : model.parameters()) {
        p.setData(p.getData() - 0.1 * p.getGrad());
    }
}
```

### 視覺化計算圖
```java
import com.micrograd.viz.GraphVisualizer;

Value a = new Value(2.0);
a.setLabel("a");

Value b = new Value(-3.0);
b.setLabel("b");

Value c = a.mul(b);
c.setLabel("c");
c.backward();

// 生成 PNG 圖片（需要安裝 Graphviz）
GraphVisualizer.draw(c, "example");  // 儲存至 output/example.png
```

## 支援的運算

| 運算 | 方法 | 說明 |
|------|------|------|
| 加法 | `a.add(b)` | a + b |
| 減法 | `a.sub(b)` | a - b |
| 乘法 | `a.mul(b)` | a × b |
| 除法 | `a.div(b)` | a ÷ b |
| 冪次 | `a.pow(n)` | aⁿ |
| 指數 | `a.exp()` | eᵃ |
| tanh | `a.tanh()` | tanh(a) |
| ReLU | `a.relu()` | max(0, a) |
| 負號 | `a.neg()` | -a |

## 核心概念

### 1. Value（計算圖節點）

每個 `Value` 物件包含：
- `data`: 純量值
- `grad`: 梯度（對最終輸出的偏導數）
- `prev`: 父節點集合
- `op`: 產生此節點的運算
- `backward`: 反向傳播函數

### 2. 反向傳播

使用拓撲排序確保正確的計算順序，然後從輸出節點開始，逆向執行每個節點的 `backward` 函數。

### 3. 鏈式法則
```
dL/da = dL/dc × dc/da
```

每個運算只需要知道「局部梯度」，然後乘上「輸出梯度」即可。

## 參考資料

- [micrograd](https://github.com/karpathy/micrograd) - Andrej Karpathy 的原始 Python 實現
- [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) - Karpathy 的神經網路教學
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) - YouTube 影片講解

# Java Makemore

A character-level language model in Java.  
Inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore).

## 簡介

這是 Karpathy 的 [makemore](https://github.com/karpathy/makemore) 第一部分（Bigram）的 Java 實現，用於學習：

- 字元級語言建模（Character-level Language Modeling）
- Bigram 統計模型
- 神經網路語言模型
- Softmax 與負對數似然損失

## 專案結構
```
java-micrograd/
├── pom.xml
├── README.md
├── data/
│   └── names.txt                    # 訓練資料
└── src/
    ├── main/java/com/micrograd/
    │   ├── Main.java                # 主程式
    │   ├── data/
    │   │   ├── Vocabulary.java      # 字元詞彙表
    │   │   └── DataLoader.java      # 資料載入
    │   ├── model/
    │   │   ├── LanguageModel.java   # 模型介面
    │   │   ├── BigramCounter.java   # 計數方法
    │   │   └── BigramNeuralNet.java # 神經網路方法
    │   ├── tensor/
    │   │   ├── Tensor.java          # 2D 張量
    │   │   └── TensorOps.java       # 張量運算
    │   └── util/
    │       └── RandomUtils.java     # 隨機採樣
    └── test/java/com/micrograd/
        ├── data/
        │   └── VocabularyTest.java
        ├── model/
        │   ├── BigramCounterTest.java
        │   └── BigramNeuralNetTest.java
        └── tensor/
            └── TensorTest.java
```

## 環境需求

- **Java 17** 或以上
- **Maven 3.6** 或以上

## 資料集

下載 `names.txt` 放到 `data/` 目錄：
```bash
mkdir -p data
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

## 快速開始

### 編譯
```bash
mvn compile
```

### 執行
```bash
mvn exec:java -Dexec.mainClass="com.micrograd.Main"
```

### 測試
```bash
mvn test
```

## 兩種方法比較

### 方法一：計數統計（BigramCounter）

- 統計所有 bigram 出現次數
- 正規化為機率分佈
- 支援 Laplace smoothing

### 方法二：神經網路（BigramNeuralNet）

- 使用 one-hot 編碼輸入
- 單層線性網路 + Softmax
- 梯度下降優化

### 結果

兩種方法會學到幾乎相同的機率分佈！

| 方法 | 訓練損失 |
|------|----------|
| 計數統計 | ~2.45 |
| 神經網路 | ~2.45 |

## 核心概念

### 1. Bigram 語言模型

給定前一個字元，預測下一個字元的機率分佈。
```
P(next_char | prev_char)
```

### 2. 負對數似然損失
```
loss = -log(P(correct_char))
```

損失越低，模型預測越準確。

### 3. Softmax

將 logits 轉換為機率分佈：
```
P(i) = exp(logit_i) / Σ exp(logit_j)
```

## 生成範例

訓練後，模型可以生成看起來像名字的字串：
```
junide
janasah
p
cede
ede
...
```