#include "NodeHandler.h"
#include "power.pb.h"
#include "mutex"
#include "google/protobuf/text_format.h"
std::mutex mutex_;
void cb(power_msg::PowerBoardStamped request, power_msg::PowerBoardStamped &reply) {
    mutex_.lock();
    std::string receive;
    google::protobuf::TextFormat::PrintToString(request, &receive);
    std::cout << receive << "\n";
    mutex_.unlock();
}
int main()
{
    core::NodeHandler nh;
    core::Rate rate(10);
    core::ServiceServer<power_msg::PowerBoardStamped, power_msg::PowerBoardStamped> &power_srv = 
    nh.serviceServer<power_msg::PowerBoardStamped, power_msg::PowerBoardStamped>("power/command", cb);

    while (1)
    {
        rate.sleep();
    }
    
    return 0;
}