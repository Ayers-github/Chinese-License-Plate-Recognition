#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "net.h"
#include <iostream>
using namespace std;
//"#�������弽�����ɼ�������������³ԥ�������������Ʋ��¸�������ѧ���۰Ĺ�ʹ������0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

string plate_chars[76] = { "#","��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
                           "��", "��", "��", "��", "��", "³", "ԥ", "��", "��", "��",
                           "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
                           "��", "ѧ", "��", "��", "��", "��", "ʹ", "��", "��", "��",
                           "��",
                           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                           "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                           "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                           "W", "X", "Y", "Z"};

//��������ǹٷ��ṩ�����ڴ�ӡ�����tensor
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
    //����ģ�͵�����
    ncnn::Net net;
    //����ģ��
    //net.load_param("clf-sim.param");
    //net.load_model("clf-sim.bin");

    net.load_param("crnn.param");
    //net.load_param("lpr2d-sim.param");
    net.load_model("crnn.bin");
    //net.load_model("lpr2d-sim.bin");

    //ʹ��opencv�ԻҶ�ͼ��ȡͼƬ
    //cv::Mat img = cv::imread("C:/Users/Administrator/Desktop/test.jpg");
    cv::Mat img = cv::imread("test.jpg");
    //��ȡͼƬ�Ŀ�
    int w = img.cols;
    //��ȡͼƬ�ĸ�
    int h = img.rows;
//    cout << w << endl << h<<endl;
    cv::imshow("aa", img);
    cv::waitKey(0);

    //��OpenCV��ͼƬתΪncnn��ʽ��ͼƬ,���ҽ�ͼƬ���ŵ�224��224֮��
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 24, 94);
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 94, 24);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 168, 48);
//    0.588, 0.193
//    pretty_print(in);
    float mean[3] = { 149.94, 149.94, 149.94 };
    float norm[3] = { 0.020319,0.020319,0.020319 };
    //��ͼƬ���й�һ��,�����ع�һ����-1~1֮��
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    //�����̸߳���
    ex.set_num_threads(1);
//    cout << in.c << endl;
//    cout << in.h << endl;
//    cout << in.w << endl;
    //cout << in.d << endl;
    //��ͼƬ���뵽������,����ǰ������
    //ex.input("input.1", in);
    pretty_print(in);
    ex.input("input.1", in);
//    cout << "����1:" << in.channel(0)[1] << endl;
    ncnn::Mat feat;
    ncnn::Mat feat0;
    //��ȡ�����������
    ex.extract("108", feat);

    ncnn::Mat m = feat;
    vector<string> final_plate_str{};

    string finale_plate;
    for (int q = 0; q < m.c; q++)
    {
        float prebs[21];
        for (int x = 0; x < m.w; x++)  //����ʮ�˸�����λ��
        {
            const float* ptr = m.channel(q);
            float preb[76];
            for (int y = 0; y < m.h; y++)  //����68���ַ���λ��
            {
                //printf("%f ", ptr[x]);
                preb[y] = ptr[x];  //��18��
                ptr += m.w;
            }
            int max_num_index = max_element(preb, preb + 68) - preb;
            //cout << max_num_index << endl;
            prebs[x] = max_num_index;
            //printf("------------------------\n");
        }

        //ȥ�ظ���ȥ�հ�
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

        // ������г���lable�����ֵ����ת��Ϊ�ַ���
        string no_repeat_blank_c = "";
        for (int hh : no_repeat_blank_label) {
            //cout<<"hh:" << hh << endl;
            no_repeat_blank_c += plate_chars[hh];
        }
        cout << "��������:" << no_repeat_blank_c << endl;

        final_plate_str.push_back(no_repeat_blank_c);
        for (string hhh : final_plate_str) {
            cout << "���г���:" << hhh << endl;
            finale_plate += hhh;
        }
    }
    string str = finale_plate;
    cout << str << endl;
    return 0;
}
