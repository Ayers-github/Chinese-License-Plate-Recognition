package com.tencent.yolov5ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YoloV5Ncnn
{
    public native boolean Init(AssetManager mgr);

    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;

        public float p1x;
        public float p1y;
        public float p2x;
        public float p2y;
        public float p3x;
        public float p3y;
        public float p4x;
        public float p4y;
        public String label;
        public float prob;
        public String color;
    }

    public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);


    static {
        System.loadLibrary("yolov5ncnn");
//        System.loadLibrary("opencv_demo");
    }
}
