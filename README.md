### make project
$ mkdir build \
$ cmake .. -DCMAKE_PREFIX_PATH=where_you_install_grpc_and_core \
$ make -j4 \

### common error

ncurse defined OK conflict with grpc, so need to uncomment ncurse.h #udef OK, and move it below #define OK(1)

### compile in ROS workerspace

pull it into catkin_ws/src, and everything is same to catkin build way.

### compile without ROS

cmake .. -DBUILD_WITH_ROS=OFF -DLOCAL_PACKAGE_PATH=your-core-installed-path
