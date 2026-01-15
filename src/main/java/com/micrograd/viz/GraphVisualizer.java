package com.micrograd.viz;

import com.micrograd.engine.Value;

import java.io.*;
import java.util.*;

/**
 * 計算圖視覺化工具
 * 直接使用本地 Graphviz 命令列工具
 */
public class GraphVisualizer {

    // ==================== 內部類別 ====================

    /**
     * 邊：連接兩個 Value 節點
     */
    public static class Edge {
        public final Value from;
        public final Value to;

        public Edge(Value from, Value to) {
            this.from = from;
            this.to = to;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Edge edge = (Edge) o;
            return from == edge.from && to == edge.to;
        }

        @Override
        public int hashCode() {
            return Objects.hash(System.identityHashCode(from), System.identityHashCode(to));
        }
    }

    /**
     * 圖資料：包含所有節點和邊
     */
    public static class GraphData {
        public final Set<Value> nodes = new LinkedHashSet<>();
        public final Set<Edge> edges = new LinkedHashSet<>();
    }

    // ==================== 核心方法 ====================

    /**
     * 追蹤計算圖：從根節點遞迴收集所有節點和邊
     */
    public static GraphData trace(Value root) {
        GraphData data = new GraphData();
        buildTrace(root, data);
        return data;
    }

    private static void buildTrace(Value v, GraphData data) {
        if (!data.nodes.contains(v)) {
            data.nodes.add(v);
            for (Value child : v.getPrev()) {
                data.edges.add(new Edge(child, v));
                buildTrace(child, data);
            }
        }
    }

    /**
     * 生成 DOT 格式字串
     */
    public static String toDot(Value root) {
        GraphData data = trace(root);
        StringBuilder sb = new StringBuilder();

        // 圖的開頭設定
        sb.append("digraph G {\n");
        sb.append("    rankdir=LR;\n");
        sb.append("    bgcolor=\"white\";\n");
        sb.append("    node [fontname=\"Arial\"];\n");
        sb.append("    edge [fontname=\"Arial\"];\n");
        sb.append("\n");

        // 遍歷所有節點
        for (Value n : data.nodes) {
            String uid = getUid(n);

            // 資料節點（方框）
            String label = buildNodeLabel(n);
            sb.append(String.format("    \"%s\" [label=\"%s\", shape=record, style=filled, fillcolor=\"lightblue\"];\n",
                    uid, label));

            // 如果有運算符，建立運算節點（圓形）
            if (n.getOp() != null && !n.getOp().isEmpty()) {
                String opUid = uid + "_op";
                sb.append(String.format("    \"%s\" [label=\"%s\", shape=circle, style=filled, fillcolor=\"lightgray\", width=0.5, height=0.5];\n",
                        opUid, n.getOp()));
                // 運算節點 → 資料節點
                sb.append(String.format("    \"%s\" -> \"%s\";\n", opUid, uid));
            }
        }

        sb.append("\n");

        // 遍歷所有邊
        for (Edge e : data.edges) {
            String fromUid = getUid(e.from);
            String toUid = getUid(e.to);

            // 如果目標節點有運算符，連到運算節點
            if (e.to.getOp() != null && !e.to.getOp().isEmpty()) {
                String toOpUid = toUid + "_op";
                sb.append(String.format("    \"%s\" -> \"%s\";\n", fromUid, toOpUid));
            } else {
                sb.append(String.format("    \"%s\" -> \"%s\";\n", fromUid, toUid));
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    /**
     * 建立節點標籤
     */
    private static String buildNodeLabel(Value n) {
        String labelPart = (n.getLabel() != null && !n.getLabel().isEmpty())
                ? n.getLabel()
                : "";

        // 格式：label | data | grad
        if (!labelPart.isEmpty()) {
            return String.format("{ %s | data: %.4f | grad: %.4f }",
                    labelPart, n.getData(), n.getGrad());
        } else {
            return String.format("{ data: %.4f | grad: %.4f }",
                    n.getData(), n.getGrad());
        }
    }

    /**
     * 取得節點的唯一識別碼
     */
    private static String getUid(Value v) {
        return "node_" + System.identityHashCode(v);
    }

    // ==================== 輸出方法（使用命令列）====================

    /**
     * 渲染成 PNG 檔案
     */
    public static void renderToPng(Value root, String filepath) throws IOException {
        renderToFile(root, filepath, "png");
    }

    /**
     * 渲染成 SVG 檔案
     */
    public static void renderToSvg(Value root, String filepath) throws IOException {
        renderToFile(root, filepath, "svg");
    }

    /**
     * 渲染成 PDF 檔案
     */
    public static void renderToPdf(Value root, String filepath) throws IOException {
        renderToFile(root, filepath, "pdf");
    }

    /**
     * 渲染成指定格式的檔案（使用本地 dot 命令）
     */
    public static void renderToFile(Value root, String filepath, String format) throws IOException {
        String dot = toDot(root);

        // 建立暫存的 DOT 檔案
        File dotFile = File.createTempFile("graph_", ".dot");
        dotFile.deleteOnExit();

        // 寫入 DOT 內容
        try (FileWriter writer = new FileWriter(dotFile)) {
            writer.write(dot);
        }

        // 確保輸出目錄存在
        File outputFile = new File(filepath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        // 執行 dot 命令
        ProcessBuilder pb = new ProcessBuilder(
                "dot",
                "-T" + format,
                dotFile.getAbsolutePath(),
                "-o",
                outputFile.getAbsolutePath()
        );

        pb.redirectErrorStream(true);

        try {
            Process process = pb.start();

            // 讀取輸出（包含錯誤訊息）
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            int exitCode = process.waitFor();

            if (exitCode == 0) {
                System.out.println("圖片已儲存至: " + filepath);
            } else {
                System.err.println("dot 命令輸出: " + output);
                throw new IOException("dot 命令執行失敗，錯誤碼: " + exitCode);
            }

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("dot 命令被中斷", e);
        } finally {
            // 刪除暫存檔
            dotFile.delete();
        }
    }

    /**
     * 儲存 DOT 原始碼
     */
    public static void saveDotFile(Value root, String filepath) throws IOException {
        try (FileWriter writer = new FileWriter(filepath)) {
            writer.write(toDot(root));
            System.out.println("DOT 檔案已儲存至: " + filepath);
        }
    }

    /**
     * 在控制台印出 DOT 原始碼
     */
    public static void printDot(Value root) {
        System.out.println(toDot(root));
    }

    // ==================== 便利方法 ====================

    /**
     * 快速繪製並開啟圖片
     */
    public static void draw(Value root, String name) {
        try {
            // 確保 output 資料夾存在
            new File("output").mkdirs();

            String filepath = "output/" + name + ".png";
            renderToPng(root, filepath);

            // 嘗試用系統預設程式開啟圖片
            if (java.awt.Desktop.isDesktopSupported()) {
                java.awt.Desktop.getDesktop().open(new File(filepath));
            }

        } catch (IOException e) {
            System.err.println("繪圖失敗: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 快速繪製（不自動開啟）
     */
    public static void drawToFile(Value root, String name) {
        try {
            new File("output").mkdirs();
            String filepath = "output/" + name + ".png";
            renderToPng(root, filepath);

        } catch (IOException e) {
            System.err.println("繪圖失敗: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 檢查 Graphviz 是否已安裝
     */
    public static boolean isGraphvizInstalled() {
        try {
            ProcessBuilder pb = new ProcessBuilder("dot", "-version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (Exception e) {
            return false;
        }
    }
}