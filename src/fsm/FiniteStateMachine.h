#ifndef FSM_H
#define FSM_H
#include "math.h"
#include <thread>
enum Mode{
    STOP = 0,
    STANCE = 1,
    WALK = 2,
    TROT = 3,
};

class FiniteStateMachine
{
    private:
        bool contact[4] = {false, false, false, false};
        double swing_counts[4] = {0, };
        const uint8_t trigger_round[4] = {1, 3, 2, 4};
        uint64_t count = 0;
        std::mutex mutex_;
        uint64_t timer_trigger_time = 0;
        int timer_count = 0;
        double freq = 1000.0;
    public:
        FiniteStateMachine(double loop_freq) : freq(loop_freq){}
        template<class M>
        void update(M mode, uint64_t swing_count) 
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            uint8_t index = trigger_round[timer_trigger_time % 4] - 1;
            timer_count -- ;
            if (timer_count < 0) 
            {
                switch(mode)
                {
                    case STOP:
                    {
                        break;
                    }
                    case WALK:
                    {
                        swing_counts[index] = swing_count;
                        timer_count = swing_count;
                        timer_trigger_time ++;
                        break;
                    }
                    case STANCE:
                    {
                        break;
                    }
                    case TROT:
                    {
                        swing_counts[index] = swing_count;
                        timer_count = index < 2? 0 : swing_count;
                        timer_trigger_time ++;
                        break;
                    }
                }
            }
            for (int i = 0; i < 4; i++)
            {
                swing_counts[i] --;
                contact[i] = swing_counts[i] > 0? true: false; // > 0 : swing; <= 0 : stance
            }
            count ++;
        }
        void swing_duration(double data[4])
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            for (int i = 0; i < 4; i++) data[i] = swing_counts[i]; 
        }
        void is_contact(bool data[4])
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            for (int i = 0; i < 4; i++) data[i] = contact[i]; 
        }
        uint64_t loop_count() 
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            return count;
        }
};

#endif