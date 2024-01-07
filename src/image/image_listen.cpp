#include <fstream>
#include "img_utils.hpp"
#include "NodeHandler.h"
std::mutex mutex_;
doc_msg::Image image;
void image_cb(doc_msg::Image msg) {
    mutex_.lock();
    image = msg;
    mutex_.unlock();
}

int main() {
    core::NodeHandler nh;
    core::Rate rate(30);
    core::Subscriber<doc_msg::Image> &img_sub = nh.subscribe<doc_msg::Image>("image", 30, image_cb);
    cv::Mat receivedImage;
    bool ret;
    std::cout << "start\n";
    while (1) {
        core::spinOnce();
        std::cout << "loop\n";
        mutex_.lock();
        decodeImageToMat(image, receivedImage, ret);
        std::cout << "decode...\n";
        if (ret) {
            std::cout << "show...\n";
            cv::imshow("Send Image", receivedImage);
            cv::waitKey(3);
        }
        mutex_.unlock();
        rate.sleep();
    }
    return 0;
}
