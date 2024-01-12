#ifndef __FSM_H
#define __FSM_H

#include <vector>
#include <leg_module.hpp>
#include <theta_error.hpp>
#include <fstream>
#include <unistd.h>
class ModeFsm
{
public:
  /* pass modules vector by reference*/
  ModeFsm(std::vector<LegModule> *module_list_);
  ModeFsm()
  {
  }

  Mode workingMode_;
  Mode prev_workingMode_;

  std::vector<LegModule> *modules_list_;

  bool if_switch_mode_msg_sent_;
  bool if_switch_mode_printed_;

  bool hall_calibrated;

  void runFsm();
  bool switchMode(Mode next_mode);
};

#endif