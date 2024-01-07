#ifndef IMG_UTILS
#define IMG_UTILS

#include <opencv2/opencv.hpp>
#include "doc.pb.h"

void encodeMatToImage(const cv::Mat& mat, doc_msg::Image& image, const std::string name) {
    image.set_width(mat.cols);
    image.set_height(mat.rows);
    image.set_name(name);
    std::cout << mat.cols << "\t" << mat.rows << "\n";
    const uchar* data = mat.ptr<uchar>(0);
    std::string imageData(data, data + mat.total() * mat.elemSize());
    image.set_data(imageData);
}

void decodeImageToMat(const doc_msg::Image& image, cv::Mat& mat, bool &ret) {
    int width = image.width();
    int height = image.height();
    std::cout << width << "\t" << height << "\n";
    ret = false;
    if (width && height) ret = true;
    else return;
    std::cout << "assign pointer\n";
    const std::string& imageData = image.data();

    mat = cv::Mat(height, width, CV_8UC3);
    uchar* matData = mat.ptr<uchar>(0);
    std::cout << "assign data\n";
    memcpy(matData, imageData.c_str(), imageData.size());
    
}

#endif