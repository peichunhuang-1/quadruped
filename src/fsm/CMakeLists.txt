add_executable(FSM_node FSM_FootholdPlanner.cpp)
target_link_libraries(FSM_node ${_CORE_LIBRARIES} robot_proto geometry_proto)

add_executable(FSM_listen_node FSM_listen.cpp)
target_link_libraries(FSM_listen_node ${_CORE_LIBRARIES} robot_proto geometry_proto LegModel LinkLegModel Eigen3::Eigen motor_proto)

add_executable(FSM_node_webot FSM_FootholdPlanner.cpp)
target_link_libraries(FSM_node_webot ${_CORE_LIBRARIES} robot_proto geometry_proto)
target_compile_definitions(FSM_node_webot PRIVATE -DSIMULATION)

add_executable(FSM_listen_node_webot FSM_listen.cpp)
target_link_libraries(FSM_listen_node_webot ${_CORE_LIBRARIES} robot_proto geometry_proto LegModel LinkLegModel Eigen3::Eigen motor_proto)
target_compile_definitions(FSM_listen_node_webot PRIVATE -DSIMULATION)