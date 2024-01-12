#include <console.hpp>
#include <yaml.h>
#include <fpga_handler.hpp>
#include <fsm.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <mutex>
#include "case_enum.hpp"
#include "angle_convert.hpp"

// Node setup
#include <NodeHandler.h>
#include <sys/time.h>

#include "motor.pb.h"
#include "power.pb.h"

#define CONFIG_PATH "/home/admin/quadruped/src/fpga_driver/config/config.yaml"

volatile sig_atomic_t sys_stop;
void inthand(int signum);

class Corgi
{
public:
  Corgi();
  void load_config_();

  YAML::Node yaml_node_;
  int modules_num_;

  FpgaHandler fpga_;
  Console console_;

  Scenario scenario_;
  Command_type cmd_type_;

  ModeFsm fsm_;
  std::vector<LegModule> modules_list_;
  std::vector<bool> powerboard_state_;
  std::mutex main_mtx_;

  std::ofstream MSG_Stream;

  // header msg
  struct timeval t_stamp;
  int seq;

  int main_irq_period_us_;
  int can_irq_period_us_;

  int max_timeout_cnt_;
  int timeout_cnt_;

  bool NO_SWITCH_TIMEDOUT_ERROR_;
  bool NO_CAN_TIMEDOUT_ERROR_;
  bool HALL_CALIBRATED_;

  bool digital_switch_;
  bool signal_switch_;
  bool power_switch_;
  bool stop_;

  void interruptHandler(core::ServiceServer<power_msg::PowerBoardStamped, power_msg::PowerBoardStamped> &power_srv, core::Subscriber<motor_msg::MotorStamped> &cmd_sub_, core::Publisher<motor_msg::MotorStamped> &state_pub_);

  void powerboardPack();

  void mainLoop_(core::ServiceServer<power_msg::PowerBoardStamped, power_msg::PowerBoardStamped> &power_srv, core::Subscriber<motor_msg::MotorStamped> &cmd_sub_, core::Publisher<motor_msg::MotorStamped> &state_pub_);
  void canLoop_();

  std::string log_path;
  std::ofstream log_stream;
};
