using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System;
using System.IO;
using Unity.Barracuda;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using System.Collections;
using System.Threading.Tasks;
using System.Collections.Generic;

// Webカメラ
public class WebCam : MonoBehaviour
{
    // カメラ
    RawImage rawImage; // RawImage
    WebCamTexture webCamTexture; //Webカメラテクスチャ

    // 描画
    Texture2D lineTexture; // ラインテクスチャ
    GUIStyle guiStyle; // GUIスタイル

    // 情報
    IList<BoundingBox> boxes; // 検出したバウンディングボックス
    float shiftX = 512f-200f; // 描画先のX座標
    float shiftY = 384f-200f; // 描画先のY座標
    float scaleFactor = 400f/(float)Detector.IMAGE_SIZE; // 描画先のスケール


    // 推論
    public Detector detector; // 物体検出
    bool isWorking = false; // 処理中

    // スタート時に呼ばれる
    void Start ()
    {
        // Webカメラの開始
        this.rawImage = GetComponent<RawImage>();
        this.webCamTexture = new WebCamTexture();
        this.webCamTexture = new WebCamTexture(
            Detector.IMAGE_SIZE, Detector.IMAGE_SIZE, 30);
        this.rawImage.texture = this.webCamTexture;
        this.webCamTexture.Play();

        // ラインテクスチャ
        this.lineTexture = new Texture2D(1, 1);
        this.lineTexture.SetPixel(0, 0, Color.red);
        this.lineTexture.Apply();

        // GUIスタイル
        this.guiStyle = new GUIStyle();
        this.guiStyle.fontSize = 50;
        this.guiStyle.normal.textColor = Color.red;
    }

    // フレーム毎に呼ばれる
    private void Update()
    {
        // 物体検出
        TFDetect();
    }

    // 物体検出
    private void TFDetect()
    {
        if (this.isWorking)
        {
            return;
        }

        this.isWorking = true;

        // 画像の前処理
        StartCoroutine(ProcessImage(result =>
        {
            // 推論の実行
            StartCoroutine(this.detector.Predict(result, boxes =>
            {
                if (boxes.Count == 0)
                {
                    this.isWorking = false;
                    return;
                }
                this.boxes = boxes;

                // 未使用のアセットをアンロード
                Resources.UnloadUnusedAssets();
                this.isWorking = false;
            }));
        }));
    }

    // 画像の前処理
    private IEnumerator ProcessImage(System.Action<Color32[]> callback)
    {
        // 画像のクロップ（WebCamTexture → Texture2D）
        yield return StartCoroutine(CropSquare(webCamTexture, texture =>
            {
                // 画像のスケール（Texture2D → Texture2D）
                var scaled = Scaled(texture,
                    Detector.IMAGE_SIZE,
                    Detector.IMAGE_SIZE);

                // コールバックを返す
                callback(scaled.GetPixels32());
            }));
    }

    // 画像のクロップ（WebCamTexture → Texture2D）
    public static IEnumerator CropSquare(WebCamTexture texture, System.Action<Texture2D> callback)
    {
        // Texture2Dの準備
        var smallest = texture.width < texture.height ? texture.width : texture.height;
        var rect = new Rect(0, 0, smallest, smallest);
        Texture2D result = new Texture2D((int)rect.width, (int)rect.height);

        // 画像のクロップ
        if (rect.width != 0 && rect.height != 0)
        {
            result.SetPixels(texture.GetPixels(
                Mathf.FloorToInt((texture.width - rect.width) / 2),
                Mathf.FloorToInt((texture.height - rect.height) / 2),
                Mathf.FloorToInt(rect.width),
                Mathf.FloorToInt(rect.height)));
            yield return null;
            result.Apply();
        }

        yield return null;
        callback(result);
    }

    // 画像のスケール（Texture2D → Texture2D）
    public static Texture2D Scaled(Texture2D texture, int width, int height)
    {
        // リサイズ後のRenderTextureの生成
        var rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(texture, rt);

        // リサイズ後のTexture2Dの生成
        var preRT = RenderTexture.active;
        RenderTexture.active = rt;
        var ret = new Texture2D(width, height);
        ret.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        ret.Apply();
        RenderTexture.active = preRT;
        RenderTexture.ReleaseTemporary(rt);
        return ret;
    }

    // GUIの表示
    public void OnGUI()
    {
        if (this.boxes != null)
        {
            foreach (var box in this.boxes)
            {
                DrawBoundingBox(box, scaleFactor, shiftX, shiftY);
            }
        }
    }

    // バウンディングボックスの描画
    void DrawBoundingBox(BoundingBox box, float scaleFactor, float shiftX, float shiftY)
    {
        var x = box.Rect.x * scaleFactor + shiftX;
        var width = box.Rect.width * scaleFactor;
        var y = box.Rect.y * scaleFactor + shiftY;
        var height = box.Rect.height * scaleFactor;
        DrawRectangle(new Rect(x, y, width, height), 4, Color.red);
        DrawLabel(new Rect(x + 10, y + 10, 200, 20), $"{box.Label}: {(int)(box.Confidence * 100)}%");
    }

    // ラベルの描画
    void DrawLabel(Rect pos, string text)
    {
        GUI.Label(pos, text, this.guiStyle);
    }

    // 矩形の描画
    void DrawRectangle(Rect area, int frameWidth, Color color)
    {
        Rect lineArea = area;
        lineArea.height = frameWidth;
        GUI.DrawTexture(lineArea, lineTexture);
        lineArea.y = area.yMax - frameWidth;
        GUI.DrawTexture(lineArea, lineTexture);
        lineArea = area;
        lineArea.width = frameWidth;
        GUI.DrawTexture(lineArea, lineTexture);
        lineArea.x = area.xMax - frameWidth;
        GUI.DrawTexture(lineArea, lineTexture);
    }
}