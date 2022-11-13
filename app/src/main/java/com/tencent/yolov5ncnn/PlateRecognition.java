package com.tencent.yolov5ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class PlateRecognition {
    private YoloV5Ncnn yolov5ncnn = new YoloV5Ncnn();
    public boolean init(AssetManager mgr){
        boolean yolo_code = yolov5ncnn.Init(mgr);
        return yolo_code;
    }

    public int floatToInt(float f){
        int i = 0;
        if(f>0) //正数
        {
            i = (int)(f*10 + 5)/10;
        }
        else if(f<0) //负数
        {
            i =  (int)(f*10 - 5)/10;
        }
        else {
            i = 0;
        }
        return i;
    }
    public YoloV5Ncnn.Obj[] detect(Bitmap bitmap, boolean use_gpu){
        YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(bitmap, use_gpu);

        return objects;
    }}
//    public static void main(String[] args) {
////        boolean ret_init = plr.init(getAssets());
//        System.out.println(1);
//    }
//}




