add_executable(cpg_node cpg.cpp cpg.hpp)
target_link_libraries(cpg_node Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto)

add_executable(cpg_node_webot cpg.cpp cpg.hpp)
target_link_libraries(cpg_node_webot Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto)
target_compile_definitions(cpg_node_webot PRIVATE -DSIMULATION)

add_library(Kuramoto Kuramoto.cpp)
target_link_libraries(Kuramoto Eigen3::Eigen) 

add_executable(Kuramoto_test main.cpp)
target_link_libraries(Kuramoto_test Kuramoto Eigen3::Eigen LinkLegModel LegModel ${_CORE_LIBRARIES} robot_proto geometry_proto motor_proto sensor_proto) 

add_executable(Kuramoto_test_webot main.cpp)
target_link_libraries(Kuramoto_test_webot Kuramoto Eigen3::Eigen LinkLegModel LegModel ${_CORE_LIBRARIES} robot_proto geometry_proto motor_proto sensor_proto )
target_compile_definitions(Kuramoto_test_webot PRIVATE -DSIMULATION)
