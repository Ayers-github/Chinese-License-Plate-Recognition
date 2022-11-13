
cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220729-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv_m/sdk/native/jni)
find_package(OpenCV REQUIRED)

#add_library(yolov5ncnn SHARED yolov5ncnn_jnitest.cpp opencv_demo.cpp)
add_library(yolov5ncnn SHARED yolov5ncnn_jnitest.cpp)

target_link_libraries(yolov5ncnn
        ncnn
        ${OpenCV_LIBS}
        jnigraphics
        )