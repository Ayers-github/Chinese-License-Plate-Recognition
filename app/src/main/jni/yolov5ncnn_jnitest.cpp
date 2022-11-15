// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <iostream>
using namespace std;
// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)
static ncnn::Net yolov5;
static ncnn::Net crnn;
static ncnn::Net color_net;
//static string plate_chars[68] = { "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
//                           "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
//                           "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
//                           "新",
//                           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
//                           "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
//                           "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
//                           "W", "X", "Y", "Z", "I", "O", "-"
//};

// crnn使用
static string plate_chars[76] = { "#","京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                                  "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                                  "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                                  "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航",
                                  "深",
                                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                  "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                                  "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                                  "W", "X", "Y", "Z"};
#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)
static string crnn_rec(const cv::Mat& bgr){

    cv::Mat img = bgr;
    //获取图片的宽
    int w = img.cols;
    //获取图片的高
    int h = img.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 168, 48);
    float mean[3] = { 149.94, 149.94, 149.94 };
    float norm[3] = { 0.020319,0.020319,0.020319 };
    //对图片进行归一化,将像素归一化到-1~1之间
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = crnn.create_extractor();
    ex.set_light_mode(true);
    //设置线程个数
    ex.set_num_threads(1);
    //将图片放入到网络中,进行前向推理
    ex.input("input.1", in);
    ncnn::Mat feat;

    //获取网络的输出结果
    ex.extract("108", feat);

    ncnn::Mat m = feat;
    vector<string> final_plate_str{};

    string finale_plate;
    for (int q = 0; q < m.c; q++)
    {
        float prebs[21];
        for (int x = 0; x < m.w; x++)  //遍历十八个车牌位置
        {
            const float* ptr = m.channel(q);
            float preb[78];
            for (int y = 0; y < m.h; y++)  //遍历68个字符串位置
            {
                preb[y] = ptr[x];  //将18个
                ptr += m.w;
            }
            int max_num_index = max_element(preb + 0, preb + 78) - preb;
//            cout<<"max_num_index"<<max_num_index<<endl;
            prebs[x] = max_num_index;
        }

        //去重复、去空白a
        vector<int> no_repeat_blank_label{};
        int pre_c = prebs[0];
        cout<<"pre_c"<<pre_c<<endl;
        if (pre_c != 0) {
            no_repeat_blank_label.push_back(pre_c);
        }
        for (int value : prebs)
        {
            if (value == 0 or value==pre_c) {
                if (value == 0 or value == pre_c) {
                    pre_c = value;
                }
                continue;
            }
            no_repeat_blank_label.push_back(value);
            pre_c = value;
        }

        // 下面进行车牌lable按照字典进行转化为字符串
        string no_repeat_blank_c = "";
        for (int hh : no_repeat_blank_label) {
            no_repeat_blank_c += plate_chars[hh];
        }
        cout << "单个车牌:" << no_repeat_blank_c << endl;

        final_plate_str.push_back(no_repeat_blank_c);
        for (string plate_char : final_plate_str) {
            cout << "所有车牌:" << plate_char << endl;
            finale_plate += plate_char;
        }
    }
    string str = finale_plate;
    cout << str << endl;
    return str;
}
static int color_rec_1(const cv::Mat& bgr){
//    ncnn::Net color_net;
//
//    color_net.load_param("color-sim.param");
//    color_net.load_model("color-sim.bin");
    //获取图片的宽
    int w = bgr.cols;
    //获取图片的高
    int h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 34, 9);

    const float norm_vals[3] = { 1.f / (0.2569 * 255), 1.f / (0.2478 * 255), 1.f / (0.2174 * 255)};

    const float mean_vals[3] = { 0.4243f * 255.f, 0.4947f * 255.f, 0.434f * 255.f};

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = color_net.create_extractor();
    //将图片放入到网络中,进行前向推理
    ex.input("input.1", in);
    ncnn::Mat feat_color;
    ex.extract("14", feat_color);
    ncnn::Mat m = feat_color;
    float color_result[3];
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {

            for (int x = 0; x < m.w; x++)
            {
                printf("%f ", ptr[x]);
                //cout << "1111:" << ptr[x];
                color_result[x] = ptr[x];
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
    int color_code = max_element(color_result, color_result + 3) - color_result;

    return color_code;
}
bool BitmapToMatrix(JNIEnv * env, jobject obj_bitmap, cv::Mat & matrix) {
    void * bitmapPixels;                                            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;                                   // Save picture parameters

    ASSERT_FALSE( AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);        // Get picture parameters
    ASSERT_FALSE( bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                  || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565 );          // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE( AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0 );  // Get picture pixels (lock memory block)
    ASSERT_FALSE( bitmapPixels );

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(matrix);                                                         // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix,matrix,cv::COLOR_RGB2BGR);

    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock
    return true;
}
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    float x;
    float y;
    float w;
    float h;
    string label;
    string color;
    float p1x;
    float p1y;
    float p2x;
    float p2y;
    float p3x;
    float p3y;
    float p4x;
    float p4y;

    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{

    /***************************************************************
      *  @brief     函数作用：未知
      *  @param     参数：
        const ncnn::Mat& anchors：即自定义设置的锚框,
        int stride：步长如8、16、32
        const ncnn::Mat& in_pad：输入的mat
        const ncnn::Mat& feat_blob：输出的mat
        float prob_threshold：未知
        std::vector<Object>& objects：处理结果存放的vector
      *  @note      备注
      *  @Sample usage:     函数的使用方法
     **************************************************************/

    const int num_grid = feat_blob.h;  //获取输出mat的h值，此处为3840=48*80,此处考虑的是stride为8的值

    int num_grid_x;
    int num_grid_y;

//    按照stride缩放，由宽和高相对大小对决定对基于w还是h缩放，最后将缩放后的w和h赋值给x和y
//    这也是为什么输入一定会处理为32的倍数的原因
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;  // 这里是w更小，输入的w为384，h为640，故num_grid_y=640/8=80，num_grid_x=48*80/80=48
        num_grid_x = num_grid / num_grid_y;
    }
//    cout<<"num_grid_x："<<num_grid_x<<endl;
//    cout<<"num_grid_y："<<num_grid_y<<endl;
    const int num_class = feat_blob.w - 13;  // 这里w等于14，减去前面四个xywh，以及conf还有四个点的8个坐标一共13个，剩下的就是类别数
    const int num_anchors = anchors.w / 2;  // anchors的数量等于anchors的w来除以2，这里的anchors的w为6，则num_anchors为3
//  torch.Size([1, 3, 80, 48, 14]),stride为8时的结果，经过conv之后的结果
    for (int q = 0; q < num_anchors; q++)  // 遍历3，即torch.Size([1, 3, 80, 48, 14])中的第2维度
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);  // 获取某个stride的三个channel之一

        for (int i = 0; i < num_grid_y; i++)  // 遍历80，即torch.Size([1, 3, 80, 48, 14])中的第三维度
        {
            for (int j = 0; j < num_grid_x; j++) // 遍历48，即torch.Size([1, 3, 80, 48, 14])中的第四维度
            {
                const float* featptr = feat.row(i * num_grid_x + j); // 对torch.Size([1, 3, 80, 48, 14])中的第四维度中的48遍历获取其值，其值应该是一个数组，包含14个数
                float box_confidence = sigmoid(featptr[4]);  // 将这个数组中的第四也就是实际第五个的conf进行sigmoid，变成0-1，赋值给锚框的置信度
                if (box_confidence >= prob_threshold)  // 判断置信度是否大于预设值，只有大于的才会进入到结果中
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + 8 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);  // 整体置信度阈值，也就是锚框置信度*类别置信度
                    if (confidence >= prob_threshold)
                    {

                        // 这里是只对xywh做了sigmoid，其中类别conf在上面已经做过了，即：float confidence = box_confidence * sigmoid(class_score);
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float p1x = featptr[5];
                        float p1y = featptr[6];
                        float p2x = featptr[7];
                        float p2y = featptr[8];
                        float p3x = featptr[9];
                        float p3y = featptr[10];
                        float p4x = featptr[11];
                        float p4y = featptr[12];

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;


                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;
                        // # landmark的进一步处理
                        p1x = p1x * anchor_w + j * stride;
                        p1y = p1y * anchor_h + i * stride;
                        p2x = p2x * anchor_w + j * stride;
                        p2y = p2y * anchor_h + i * stride;
                        p3x = p3x * anchor_w + j * stride;
                        p3y = p3y * anchor_h + i * stride;
                        p4x = p4x * anchor_w + j * stride;
                        p4y = p4y * anchor_h + i * stride;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.x = x0;
                        obj.y = y0;
                        obj.w = x1 - x0;
                        obj.h = y1 - y0;
                        obj.label = "";
                        obj.color = "";
                        obj.prob = confidence;
                        obj.p1x = p1x;
                        obj.p1y = p1y;
                        obj.p2x = p2x;
                        obj.p2y = p2y;
                        obj.p3x = p3x;
                        obj.p3y = p3y;
                        obj.p4x = p4x;
                        obj.p4y = p4y;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID p1xId;
static jfieldID p1yId;
static jfieldID p2xId;
static jfieldID p2yId;
static jfieldID p3xId;
static jfieldID p3yId;
static jfieldID p4xId;
static jfieldID p4yId;

static jfieldID labelId;
static jfieldID probId;
static jfieldID colorId;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolov5.opt = opt;

    yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    // init param
    {
//        int ret = yolov5.load_param(mgr, "yolov5s.param");
        int ret = yolov5.load_param(mgr, "best.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_param failed");
            return JNI_FALSE;
        }
        else{
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_param success!");
        }
    }

    // init bin
    {
//        int ret = yolov5.load_model(mgr, "yolov5s.bin");
        int ret = yolov5.load_model(mgr, "best.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yolov5ncnn/YoloV5Ncnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolov5ncnn/YoloV5Ncnn;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    p1xId = env->GetFieldID(objCls, "p1x", "F");
    p1yId = env->GetFieldID(objCls, "p1y", "F");
    p2xId = env->GetFieldID(objCls, "p2x", "F");
    p2yId = env->GetFieldID(objCls, "p2y", "F");
    p3xId = env->GetFieldID(objCls, "p3x", "F");
    p3yId = env->GetFieldID(objCls, "p3y", "F");
    p4xId = env->GetFieldID(objCls, "p4x", "F");
    p4yId = env->GetFieldID(objCls, "p4y", "F");

    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");
    colorId = env->GetFieldID(objCls, "color", "Ljava/lang/String;");

    // TODO: implement Init()
    ncnn::Option opt_crnn;
    opt_crnn.lightmode = true;
    opt_crnn.num_threads = 4;
    opt_crnn.blob_allocator = &g_blob_pool_allocator;
    opt_crnn.workspace_allocator = &g_workspace_pool_allocator;
    opt_crnn.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt_crnn.use_vulkan_compute = true;

    crnn.opt = opt_crnn;

    // init param
    {
        int ret = crnn.load_param(mgr, "crnn.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "crnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = crnn.load_model(mgr, "crnn.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "crnn", "load_model failed");
            return JNI_FALSE;
        } else{
            __android_log_print(ANDROID_LOG_DEBUG, "crnn", "load_model success!");
        }
    }

    // TODO: implement Init()
    ncnn::Option opt_color;
    opt_color.lightmode = true;
    opt_color.num_threads = 4;
    opt_color.blob_allocator = &g_blob_pool_allocator;
    opt_color.workspace_allocator = &g_workspace_pool_allocator;
    opt_color.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt_color.use_vulkan_compute = true;

    color_net.opt = opt_color;

    // init param
    {
        int ret = color_net.load_param(mgr, "color-sim.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "color_classify", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = color_net.load_model(mgr, "color-sim.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "color_classify", "load_model failed");
            return JNI_FALSE;
        } else{
            __android_log_print(ANDROID_LOG_DEBUG, "color_classify", "load_model success!");
        }
    }

    return JNI_TRUE;
}

// public native Obj[] regognition(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    const int target_size = 640;

    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    std::vector<Object> objects;
    {
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov5.create_extractor();

        ex.set_vulkan_compute(use_gpu);

//        ex.input("images", in_pad);
        ex.input("data", in_pad);

        std::vector<Object> proposals;

        // stride 8
        {
            ncnn::Mat out;
            ex.extract("stride_8", out);
            ncnn::Mat anchors(6);
            anchors[0] = 4.f;
            anchors[1] = 5.f;
            anchors[2] = 8.f;
            anchors[3] = 10.f;
            anchors[4] = 13.f;
            anchors[5] = 16.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
            proposals.insert(proposals.end(), objects8.begin(), objects8.end());

        }
        //
        // stride 16
        {
            ncnn::Mat out;
            //ex.extract("781", out);
            ex.extract("stride_16", out);

            ncnn::Mat anchors(6);
            anchors[0] = 23.f;
            anchors[1] = 29.f;
            anchors[2] = 43.f;
            anchors[3] = 55.f;
            anchors[4] = 73.f;
            anchors[5] = 105.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            //ex.extract("801", out);
            ex.extract("stride_32", out);
            ncnn::Mat anchors(6);
            anchors[0] = 146.f;
            anchors[1] = 217.f;
            anchors[2] = 231.f;
            anchors[3] = 300.f;
            anchors[4] = 335.f;
            anchors[5] = 433.f;
            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++) {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;

            float p1x = (objects[i].p1x - (wpad / 2)) / scale;
            float p1y = (objects[i].p1y - (hpad / 2)) / scale;
            float p2x = (objects[i].p2x - (wpad / 2)) / scale;
            float p2y = (objects[i].p2y - (hpad / 2)) / scale;
            float p3x = (objects[i].p3x - (wpad / 2)) / scale;
            float p3y = (objects[i].p3y - (hpad / 2)) / scale;
            float p4x = (objects[i].p4x - (wpad / 2)) / scale;
            float p4y = (objects[i].p4y - (hpad / 2)) / scale;

            float x1 = (objects[i].x + objects[i].w- (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

            p1x = std::max(std::min(p1x, (float) (width - 1)), 0.f);
            p1y = std::max(std::min(p1y, (float) (height - 1)), 0.f);
            p2x = std::max(std::min(p2x, (float) (width - 1)), 0.f);
            p2y = std::max(std::min(p2y, (float) (height - 1)), 0.f);
            p3x = std::max(std::min(p3x, (float) (width - 1)), 0.f);
            p3y = std::max(std::min(p3y, (float) (height - 1)), 0.f);
            p4x = std::max(std::min(p4x, (float) (width - 1)), 0.f);
            p4y = std::max(std::min(p4y, (float) (height - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;

            objects[i].w= x1 - x0;
            objects[i].h = y1 - y0;

            objects[i].p1x = p1x;
            objects[i].p1y = p1y;
            objects[i].p2x = p2x;
            objects[i].p2y = p2y;
            objects[i].p3x = p3x;
            objects[i].p3y = p3y;
            objects[i].p4x = p4x;
            objects[i].p4y = p4y;
        }
    }

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i=0; i<objects.size(); i++)
    {
        // letterbox pad to multiple of 32
        cv::Mat image;
        BitmapToMatrix(env, bitmap, image);
        const Object& obj = objects[i];

        float new_x1 = objects[i].p3x - objects[i].x;
        float new_y1 = objects[i].p3y - objects[i].y;
        float new_x2 = objects[i].p4x - objects[i].x;
        float new_y2 = objects[i].p4y - objects[i].y;
        float new_x3 = objects[i].p2x - objects[i].x;
        float new_y3 = objects[i].p2y - objects[i].y;
        float new_x4 = objects[i].p1x - objects[i].x;
        float new_y4 = objects[i].p1y - objects[i].y;

        cv::Point2f src_points[4];
        cv::Point2f dst_points[4];
        //通过Image Watch查看的二维码四个角点坐标
        src_points[0]=cv::Point2f(new_x1, new_y1);
        src_points[1]=cv::Point2f(new_x2, new_y2);
        src_points[2]=cv::Point2f(new_x3, new_y3);
        src_points[3]=cv::Point2f(new_x4, new_y4);
        //期望透视变换后二维码四个角点的坐标
        dst_points[0]=cv::Point2f(0.0, 0.0);
        dst_points[1]=cv::Point2f(168.0, 0.0);
        dst_points[2]=cv::Point2f(0.0, 48.0);
        dst_points[3]=cv::Point2f(168.0, 48.0);

        cv::Mat rotation,img_warp;
        cv::Rect_<float> rect;
        rect.x = objects[i].x;
        rect.y = objects[i].y;
        rect.height = objects[i].h;
        rect.width = objects[i].w;
        cv::Mat ROI = image(rect);
        rotation=getPerspectiveTransform(src_points,dst_points);
//        cout<<"image.size():"<<image.size()<<endl;
        warpPerspective(ROI,ROI,rotation,cv::Size(168, 48));

        string plate_str=crnn_rec(ROI);
        string color_names[3] = {
//                "blue", "green","yellow"
                "蓝", "绿", "黄"
        };
        int color_code = color_rec_1(ROI);
        string color_name = color_names[color_code];

        char*p=(char*)plate_str.data();
        char*p_color=(char*)color_name.data();
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].w);
        env->SetFloatField(jObj, hId, objects[i].h);
        env->SetFloatField(jObj, p1xId, objects[i].p1x);
        env->SetFloatField(jObj, p1yId, objects[i].p1y);
        env->SetFloatField(jObj, p2xId, objects[i].p2x);
        env->SetFloatField(jObj, p2yId, objects[i].p2y);
        env->SetFloatField(jObj, p3xId, objects[i].p3x);
        env->SetFloatField(jObj, p3yId, objects[i].p3y);
        env->SetFloatField(jObj, p4xId, objects[i].p4x);
        env->SetFloatField(jObj, p4yId, objects[i].p4y);

//        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetObjectField(jObj, labelId, env->NewStringUTF(p));
        env->SetObjectField(jObj, colorId, env->NewStringUTF(p_color));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect", elasped);

    return jObjArray;
    }
}