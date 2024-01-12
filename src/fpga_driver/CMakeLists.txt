set(INC_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

include_directories(${INC_DIR})
include_directories(${YAML_CPP_INCLUDE_DIR})
include_directories(${CMAKE_PREFIX_PATH}/include/yaml-cpp)
include_directories(${CURSES_INCLUDE_DIR})
include_directories(${CMAKE_PREFIX_PATH}/include/ncurses)


# add_executable(fpga_srv_main ${SRC_DIR}/fpga_server.cpp
#                              ${SRC_DIR}/leg_module.cpp
#                              ${SRC_DIR}/fpga_handler.cpp
#                              ${SRC_DIR}/color.cpp
#                              ${SRC_DIR}/fsm.cpp
#                              ${SRC_DIR}/console.cpp
#                              ${SRC_DIR}/theta_error.cpp
#                              ${SRC_DIR}/angle_convert.cpp
#                              ${SRC_DIR}/NiFpga.c
#                              ${SRC_DIR}/NiFpga_FPGA_CANBus_IMU_4module_IRQ.c)

# target_link_libraries(fpga_srv_main yaml-cpp ${CURSES_LIBRARY} -lncurses ${CMAKE_DL_LIBS} ${_CORE_LIBRARIES} motor_proto power_proto -lpthread Eigen3::Eigen)

add_executable(test_receive ${SRC_DIR}/testfpga.cpp)
target_link_libraries(test_receive ${_CORE_LIBRARIES} power_proto -lpthread)

# add_executable(force_control ${SRC_DIR}/force_control.cpp)
# target_link_libraries(force_control Eigen3::Eigen)