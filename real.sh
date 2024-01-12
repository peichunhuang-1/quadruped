# /bin/bash
trap "kill -- -$$" INT

NodeCore &
sleep .1
pushd build
./FSM_node &
sleep .1
./FSM_listen_node &
sleep .1
./imu_cx5 & 
sleep .1
# ./fpga_srv_main &
# sleep .1
popd



wait