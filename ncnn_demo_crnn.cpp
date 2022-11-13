#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "net.h"
#include <iostream>
using namespace std;
string plate_chars[76] = { "#","��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
                           "��", "��", "��", "��", "��", "³", "ԥ", "��", "��", "��",
                           "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
                           "��", "ѧ", "��", "��", "��", "��", "ʹ", "��", "��", "��",
                           "��",
                           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                           "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                           "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                           "W", "X", "Y", "Z"};

int main()
/*!
 *
 * @return
 */
{
    //����ģ�͵�����
    ncnn::Net net;

    //����ģ��
    net.load_param("crnn.param");
    net.load_model("crnn.bin");

    //ʹ��opencv�ԻҶ�ͼ��ȡͼƬ
    cv::Mat img = cv::imread("test.jpg");

    //��ȡͼƬ�Ŀ�
    int w = img.cols;

    //��ȡͼƬ�ĸ�
    int h = img.rows;

//    ��ʾͼƬ
    cv::imshow("img", img);
    cv::waitKey(0);

    //��OpenCV��ͼƬתΪncnn��ʽ��ͼƬ,���ҽ�ͼƬresize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 168, 48);

    float mean[3] = { 149.94, 149.94, 149.94 };
    float norm[3] = { 0.020319,0.020319,0.020319 };
    //��ͼƬ���й�һ��,�����ع�һ����-1~1֮����ٽ�
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    //�����̸߳���
    ex.set_num_threads(1);
    ex.input("input.1", in);
    ncnn::Mat feat;

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
                preb[y] = ptr[x];  //��18��
                ptr += m.w;
            }
            int max_num_index = max_element(preb, preb + 68) - preb;
            prebs[x] = max_num_index;
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
            no_repeat_blank_c += plate_chars[hh];
        }
        cout << "��������:" << no_repeat_blank_c << endl;

        final_plate_str.push_back(no_repeat_blank_c);
        for (string plate_char : final_plate_str) {
            cout << "���г���:" << plate_char << endl;
            finale_plate += plate_char;
        }
    }
    string str = finale_plate;
    cout << str << endl;
    return 0;
}
