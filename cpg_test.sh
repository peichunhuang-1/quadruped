# /bin/bash
trap "kill -- -$$" INT

source ~/.bash_profile

NodeCore &
sleep .1
pushd build
./cpg_node &
sleep .1
popd

pushd ./js/control_panel
node quadruped.js
popd

wait
