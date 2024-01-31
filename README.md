### make project
$ mkdir build \
$ cmake .. -DCMAKE_PREFIX_PATH=where_you_install_grpc_and_core \
$ make -j4 \

### common error

ncurse defined OK conflict with grpc, so need to uncomment ncurse.h #udef OK, and move it below #define OK(1)
