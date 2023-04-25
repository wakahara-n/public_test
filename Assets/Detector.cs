using System;
using Unity.Barracuda;
using System.Linq;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;

// 物体検出
public class Detector : MonoBehaviour
{
    // アンカー
    private float[] Anchors = new float[]
    {
        1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
    };

    // リソース
    public NNModel modelFile; // モデル
    public TextAsset labelsFile; // ラベル

    // パラメータ
    public const int IMAGE_SIZE = 416; // 画像サイズ
    private const int IMAGE_MEAN = 0;
    private const float IMAGE_STD = 1f;
    private const string INPUT_NAME = "image";
    private const string OUTPUT_NAME = "grid";

    // 出力のパース
    private const int ROW_COUNT = 13; // 行
    private const int COL_COUNT = 13; // 列
    private const int BOXES_PER_CELL = 5; // セル毎のボックス数
    private const int BOX_INFO_FEATURE_COUNT = 5; // ボックス情報の特徴数
    private const int CLASS_COUNT = 20; // クラス数
    private const float CELL_WIDTH = 32; // セル幅
    private const float CELL_HEIGHT = 32; // セル高さ

    // 出力のフィルタリング
    private const float MINIMUM_CONFIDENCE = 0.3f; // 最小検出信頼度

    // 推論
    private IWorker worker; // ワーカー
    private string[] labels; // ラベル

    // スタート時に呼ばれる
    void Start()
    {
        // ラベルとモデルの読み込み
        this.labels = Regex.Split(this.labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();
        var model = ModelLoader.Load(this.modelFile);

        // ワーカーの生成
        this.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    // 推論
    public IEnumerator Predict(Color32[] picture, System.Action<IList<BoundingBox>> callback)
    {
        // 入力テンソルの生成
        using (var tensor = TransformInput(picture, IMAGE_SIZE, IMAGE_SIZE))
        {
            // 入力の生成
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add(INPUT_NAME, tensor);

            // 推論の実行
            yield return StartCoroutine(worker.ExecuteAsync(inputs));

            // 出力の生成
            var output = worker.PeekOutput(OUTPUT_NAME);
            var results = ParseOutputs(output);
            var boxes = FilterBoundingBoxes(results, 5, MINIMUM_CONFIDENCE);

            // 結果を返す
            callback(boxes);
        }
    }

    // 入力テンソルの生成
    public static Tensor TransformInput(Color32[] pic, int width, int height)
    {
        float[] floatValues = new float[width * height * 3];
        for (int i = 0; i < pic.Length; ++i)
        {
            var color = pic[i];
            floatValues[i * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
        }
        return new Tensor(1, height, width, 3, floatValues);
    }

    // 出力のパース
    private IList<BoundingBox> ParseOutputs(Tensor output, float threshold = .3F)
    {
        var boxes = new List<BoundingBox>();
        for (int cy = 0; cy < COL_COUNT; cy++)
        {
            for (int cx = 0; cx < ROW_COUNT; cx++)
            {
                for (int box = 0; box < BOXES_PER_CELL; box++)
                {
                    var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                    // バウンディングボックスの寸法と信頼度の取得
                    var dimensions = GetBoundingBoxDimensions(output, cx, cy, channel);
                    float confidence = GetConfidence(output, cx, cy, channel);
                    if (confidence < threshold)
                    {
                        continue;
                    }

                    // スコアが最大のクラスのINDEXとスコアの取得
                    float[] predictedClasses = GetPredictedClasses(output, cx, cy, channel);
                    var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                    var topScore = topResultScore * confidence;
                    if (topScore < threshold)
                    {
                        continue;
                    }

                    // バウンディングボックスをセルにマッピング
                    var mappedBoundingBox = MapBoundingBoxToCell(cx, cy, box, dimensions);
 
                    // バウンディングボックスの追加
                    var boundingBox = new BoundingBox();
                    boundingBox.Rect = new Rect(
                        (mappedBoundingBox.x - mappedBoundingBox.width / 2),
                        (mappedBoundingBox.y - mappedBoundingBox.height / 2),
                        mappedBoundingBox.width,
                        mappedBoundingBox.height);
                    boundingBox.Confidence = topScore;
                    boundingBox.Label = labels[topResultIndex];
                    boxes.Add(boundingBox);
                }
            }
        }
        return boxes;
    }

    // バウンディングボックスの抽出
    private Rect GetBoundingBoxDimensions(Tensor output, int x, int y, int channel)
    {
        return new Rect(
            output[0, x, y, channel],
            output[0, x, y, channel + 1],
            output[0, x, y, channel + 2],
            output[0, x, y, channel + 3]);
    }

    // 信頼度の抽出
    private float GetConfidence(Tensor output, int x, int y, int channel)
    {
        return Sigmoid(output[0, x, y, channel + 4]);
    }

    // 予測クラスの抽出
    private float[] GetPredictedClasses(Tensor output, int x, int y, int channel)
    {
        float[] predictedClasses = new float[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = output[0, x, y, predictedClass + predictedClassOffset];
        }
        return Softmax(predictedClasses);
    }

    // スコアが最大のクラスのINDEXとスコアの取得
    private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
    {
        return predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();
    }

    // バウンディングボックスをセルにマッピング
    private Rect MapBoundingBoxToCell(int x, int y, int box, Rect dimensions)
    {
        return new Rect(
            ((float)y + Sigmoid(dimensions.x)) * CELL_WIDTH,
            ((float)x + Sigmoid(dimensions.y)) * CELL_HEIGHT,
            (float)Math.Exp(dimensions.width) * CELL_WIDTH * Anchors[box * 2],
            (float)Math.Exp(dimensions.height) * CELL_HEIGHT * Anchors[box * 2 + 1]);
    }

    // バウンディングボックスのフィルタリング
    private IList<BoundingBox> FilterBoundingBoxes(IList<BoundingBox> boxes, int limit, float threshold)
    {
        var activeCount = boxes.Count;
        var isActiveBoxes = new bool[boxes.Count];
        for (int i = 0; i < isActiveBoxes.Length; i++)
        {
            isActiveBoxes[i] = true;
        }
        var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
            .OrderByDescending(b => b.Box.Confidence)
            .ToList();
        var results = new List<BoundingBox>();
        for (int i = 0; i < boxes.Count; i++)
        {
            if (isActiveBoxes[i])
            {
                var boxA = sortedBoxes[i].Box;
                results.Add(boxA);
                if (results.Count >= limit)
                {
                    break;
                }
                for (var j = i + 1; j < boxes.Count; j++)
                {
                    if (isActiveBoxes[j])
                    {
                        var boxB = sortedBoxes[j].Box;
                        if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            activeCount--;
                            if (activeCount <= 0)
                            {
                                break;
                            }
                        }
                    }
                }
                if (activeCount <= 0)
                {
                    break;
                }
            }
        }
        return results;
    }

    // IoU（評価指標）の計算
    private float IntersectionOverUnion(Rect boundingBoxA, Rect boundingBoxB)
    {
        var areaA = boundingBoxA.width * boundingBoxA.height;
        if (areaA <= 0)
        {
            return 0;
        }
        var areaB = boundingBoxB.width * boundingBoxB.height;
        if (areaB <= 0)
        {
            return 0;
        }
        var minX = Math.Max(boundingBoxA.xMin, boundingBoxB.xMin);
        var minY = Math.Max(boundingBoxA.yMin, boundingBoxB.yMin);
        var maxX = Math.Min(boundingBoxA.xMax, boundingBoxB.xMax);
        var maxY = Math.Min(boundingBoxA.yMax, boundingBoxB.yMax);
        var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    // シグモイド
    private float Sigmoid(float value)
    {
        var k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    // ソフトマックス
    private float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => Math.Exp(v - maxVal));
        var sumExp = exp.Sum();
        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }
}

// バウンディングボックス
public class BoundingBox
{
    public string Label; // ラベル
    public float Confidence; // 信頼度
    public Rect Rect; //矩形
}