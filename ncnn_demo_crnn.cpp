#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "net.h"
#include <iostream>
using namespace std;
//"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

string plate_chars[76] = { "#","京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                           "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                           "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                           "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航",
                           "深",
                           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                           "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                           "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                           "W", "X", "Y", "Z"};

//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat & m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                printf("%f ", ptr[x]);
                //cout << "1111:" << ptr[x];
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}


int main()
/*!
 *
 * @return
 */
{
    //定义模型的网络
    ncnn::Net net;
    //加载模型
    //net.load_param("clf-sim.param");
    //net.load_model("clf-sim.bin");

    net.load_param("crnn.param");
    //net.load_param("lpr2d-sim.param");
    net.load_model("crnn.bin");
    //net.load_model("lpr2d-sim.bin");

    //使用opencv以灰度图读取图片
    //cv::Mat img = cv::imread("C:/Users/Administrator/Desktop/test.jpg");
    cv::Mat img = cv::imread("test.jpg");
    //获取图片的宽
    int w = img.cols;
    //获取图片的高
    int h = img.rows;
//    cout << w << endl << h<<endl;
    cv::imshow("aa", img);
    cv::waitKey(0);

    //将OpenCV的图片转为ncnn格式的图片,并且将图片缩放到224×224之间
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 24, 94);
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 94, 24);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 168, 48);
//    0.588, 0.193
//    pretty_print(in);
    float mean[3] = { 149.94, 149.94, 149.94 };
    float norm[3] = { 0.020319,0.020319,0.020319 };
    //对图片进行归一化,将像素归一化到-1~1之间
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    //设置线程个数
    ex.set_num_threads(1);
//    cout << in.c << endl;
//    cout << in.h << endl;
//    cout << in.w << endl;
    //cout << in.d << endl;
    //将图片放入到网络中,进行前向推理
    //ex.input("input.1", in);
    pretty_print(in);
    ex.input("input.1", in);
//    cout << "输入1:" << in.channel(0)[1] << endl;
    ncnn::Mat feat;
    ncnn::Mat feat0;
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
            float preb[76];
            for (int y = 0; y < m.h; y++)  //遍历68个字符串位置
            {
                //printf("%f ", ptr[x]);
                preb[y] = ptr[x];  //将18个
                ptr += m.w;
            }
            int max_num_index = max_element(preb, preb + 68) - preb;
            //cout << max_num_index << endl;
            prebs[x] = max_num_index;
            //printf("------------------------\n");
        }

        //去重复、去空白
        vector<int> no_repeat_blank_label{};
        int pre_c = prebs[0];
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
            //cout<<"hh:" << hh << endl;
            no_repeat_blank_c += plate_chars[hh];
        }
        cout << "单个车牌:" << no_repeat_blank_c << endl;

        final_plate_str.push_back(no_repeat_blank_c);
        for (string hhh : final_plate_str) {
            cout << "所有车牌:" << hhh << endl;
            finale_plate += hhh;
        }
    }
    string str = finale_plate;
    cout << str << endl;
    return 0;
}
