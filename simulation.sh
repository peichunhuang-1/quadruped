# /bin/bash
trap "kill -- -$$" INT

NodeCore &
sleep .1
echo Enter to continue...
read
pushd build
./FSM_node_webot &
sleep .1
./FSM_listen_node_webot &
sleep .1
popd
pushd ~/Desktop/quadruped/js/control_panel
node quadruped.js &
popd

wait