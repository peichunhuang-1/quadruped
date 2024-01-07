#include <iostream>
#include <fstream>
#include "img_utils.hpp"
#include "NodeHandler.h"

int main() {
    
    core::NodeHandler nh;
    core::Rate rate(30);
    core::Publisher<doc_msg::Image> &img_pub = nh.advertise<doc_msg::Image>("image");
    cv::VideoCapture cap(1);
    cv::Mat originalImage;
    if (!cap.isOpened()) {
        std::cout << "Could not read from camera!\n";
        return -1;
    }
    bool ret;
    while(1) {
        cap >> originalImage ;
        doc_msg::Image protobufImage;
        encodeMatToImage(originalImage, protobufImage, "image");
        img_pub.publish(protobufImage);
        rate.sleep();
    }

    return 0;
}
