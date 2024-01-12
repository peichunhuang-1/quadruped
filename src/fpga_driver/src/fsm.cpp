#include <fsm.hpp>

ModeFsm::ModeFsm(std::vector<LegModule> *_modules)
{
    workingMode_ = Mode::REST;
    prev_workingMode_ = Mode::REST;

    modules_list_ = _modules;
    if_switch_mode_msg_sent_ = false;
    if_switch_mode_printed_ = false;
    hall_calibrated = true;
    runFsm();
}

void ModeFsm::runFsm()
{
    switch (workingMode_)
    {
    case Mode::REST:
    {
        /* if (!if_switch_mode_printed_)
        {
            // important_message("[FSM] REST MODE RUNNING ");
            // if_switch_mode_printed_ = true;
        } */
    }
    break;

    case Mode::SET_ZERO:
    {
        /* if (!if_switch_mode_printed_)
        {
            important_message(" [FSM] SET_ZERO MODE RUNNING ");
            if_switch_mode_printed_ = true;
        } */
    }
    break;

    case Mode::HALL_CALIBRATE:
    {
        /* if (!if_switch_mode_printed_)
        {
            important_message(" [FSM] HALL_CALIBRATE MODE RUNNING ");
            if_switch_mode_printed_ = true;
        } */

        CAN_txdata txdata_zero[2];
        txdata_zero[0].position_ = 0;
        txdata_zero[0].torque_ = 0;
        txdata_zero[0].KP_ = 5;
        txdata_zero[0].KI_ = 0;
        txdata_zero[0].KD_ = 1.5;
        txdata_zero[1].position_ = 0 * M_PI / 180.0;
        txdata_zero[1].torque_ = 0;
        txdata_zero[1].KP_ = 5;
        txdata_zero[1].KI_ = 0;
        txdata_zero[1].KD_ = 1.5;

        int module_cal_finished = 1;

        while (1)
        {
            module_cal_finished = 1;
            for (int i = 0; i < 4; i++)
            {
                if (modules_list_->at(i).enable_)
                {
                    CAN_txdata idle_txdata_;
                    idle_txdata_.position_ = 0;
                    idle_txdata_.torque_ = 0;
                    idle_txdata_.KP_ = 0;
                    idle_txdata_.KI_ = 0;
                    idle_txdata_.KD_ = 0;

                    // modules_list_->at(i).CAN_timeoutCheck();

                    modules_list_->at(i).io_.CAN_send_command(idle_txdata_, idle_txdata_);

                    modules_list_->at(i).io_.CAN_recieve_feedback(&modules_list_->at(i).rxdata_buffer_[0], &modules_list_->at(i).rxdata_buffer_[1]);

                    int mod1_finished = 0;
                    int mod2_finished = 0;

                    if (modules_list_->at(i).rxdata_buffer_[0].calibrate_finish_ == 2)
                        mod1_finished = 1;
                    if (modules_list_->at(i).rxdata_buffer_[1].calibrate_finish_ == 2)
                        mod2_finished = 1;

                    module_cal_finished *= mod1_finished;
                    module_cal_finished *= mod2_finished;
                }
            }
            if (module_cal_finished == 1)
                break;
            usleep(1e4);
        }

        for (int i = 0; i < 4; i++)
        {
            if (modules_list_->at(i).enable_)
            {
                modules_list_->at(i).CAN_rx_timedout_[0] = false;
                modules_list_->at(i).CAN_rx_timedout_[1] = false;
                modules_list_->at(i).CAN_tx_timedout_[0] = false;
                modules_list_->at(i).CAN_tx_timedout_[1] = false;
                modules_list_->at(i).io_.motorR_bias = modules_list_->at(i).linkR_bias;
                modules_list_->at(i).io_.motorL_bias = modules_list_->at(i).linkL_bias;
                modules_list_->at(i).io_.CAN_set_mode(Mode::MOTOR);
            }
        }

        double dt_ = 0.01;  // second
        double vel_ = 0.25; // rad/s
        double command[4][2];
        double dir_[4][2];
        double tolerance = 0.05;

        for (int i = 0; i < 4; i++)
        {
            // LOGFILE << "Module " << i << std::endl;
            if (modules_list_->at(i).enable_)
            {
                modules_list_->at(i).io_.CAN_recieve_feedback(&modules_list_->at(i).rxdata_buffer_[0], &modules_list_->at(i).rxdata_buffer_[1]);

                command[i][0] = modules_list_->at(i).rxdata_buffer_[0].position_;
                if (theta_error(modules_list_->at(i).rxdata_buffer_[0].position_, txdata_zero[0].position_) > 0)
                    dir_[i][0] = 1;
                else
                    dir_[i][0] = -1;

                command[i][1] = modules_list_->at(i).rxdata_buffer_[1].position_;
                if (theta_error(modules_list_->at(i).rxdata_buffer_[1].position_, txdata_zero[1].position_) > 0)
                    dir_[i][1] = 1;
                else
                    dir_[i][1] = -1;
            }
        }

        while (1)
        {
            int finished = 1;

            for (int i = 0; i < 4; i++)
            {
                if (modules_list_->at(i).enable_)
                {
                    modules_list_->at(i).io_.CAN_recieve_feedback(&modules_list_->at(i).rxdata_buffer_[0], &modules_list_->at(i).rxdata_buffer_[1]);

                    CAN_txdata txdata_[2];

                    for (int j = 0; j < 2; j++)
                    {
                        double errj = 0;
                        errj = theta_error(command[i][j], txdata_zero[0].position_);
                        txdata_[j].position_ = 0;
                        txdata_[j].torque_ = 0;
                        txdata_[j].KP_ = 0;
                        txdata_[j].KI_ = 0;
                        txdata_[j].KD_ = 0;

                        if (fabs(errj) < tolerance)
                        {
                            txdata_[j].position_ = 0;
                            finished *= 1;
                        }
                        else
                        {
                            command[i][j] += dir_[i][j] * vel_ * dt_;

                            txdata_[j].position_ = command[i][j];
                            txdata_[j].torque_ = 0;
                            txdata_[j].KP_ = 50;
                            txdata_[j].KI_ = 0;
                            txdata_[j].KD_ = 1.5;
                            finished *= 0;
                        }
                    }
                    modules_list_->at(i).io_.CAN_send_command(txdata_[0], txdata_[1]);
                }
            }

            if (finished)
                break;

            usleep(dt_ * 1000 * 1000);
        }

        hall_calibrated = true;
        switchMode(Mode::MOTOR);
    }
    break;

    case Mode::MOTOR:
    {
        /* if (!if_switch_mode_printed_)
        {
            important_message(" [FSM] MOTOR MODE RUNNING ");
            if_switch_mode_printed_ = true;
        } */
    }
    break;
    }
}

bool ModeFsm::switchMode(Mode next_mode)
{

    int mode_switched_cnt = 0;
    int module_enabled = 0;
    bool success = false;
    std::ofstream log_;
    log_.open("/home/admin/fpga_driver2/log/log_fsm.txt");

    for (int i = 0; i < 4; i++)
    {
        if (modules_list_->at(i).enable_)
        {
            module_enabled++;
        }
    }

    double time_elapsed = 0;

    while (1)
    {
        if (mode_switched_cnt == module_enabled)
        {
            success = true;
            // hall_calibrate_enable_ = false;
            break;
        }
        else if (time_elapsed > 1)
        {
            /* Timeout */
            success = false;
            break;
        }
        else
            mode_switched_cnt = 0;

        for (int i = 0; i < 4; i++)
        {
            if (modules_list_->at(i).enable_)
            {
                modules_list_->at(i).io_.CAN_set_mode(next_mode);

                modules_list_->at(i).io_.CAN_recieve_feedback(&modules_list_->at(i).rxdata_buffer_[0], &modules_list_->at(i).rxdata_buffer_[1]);

                if (modules_list_->at(i).rxdata_buffer_[0].mode_ == next_mode && modules_list_->at(i).rxdata_buffer_[1].mode_ == next_mode)
                {
                    mode_switched_cnt++;
                }
            }
        }

        time_elapsed += 0.01;
        usleep(1e4);
    }

    prev_workingMode_ = workingMode_;
    workingMode_ = next_mode;

    return success;
}
