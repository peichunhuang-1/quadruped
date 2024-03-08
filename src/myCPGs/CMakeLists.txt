add_library(MyCPGs MyCPGs.cpp)
target_link_libraries(MyCPGs Eigen3::Eigen LegModel LinkLegModel) 
add_executable(my_cpg_test main.cpp)
target_link_libraries(my_cpg_test MyCPGs Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto motor_proto sensor_proto)
target_compile_definitions(my_cpg_test PRIVATE -DSIMULATION)

add_executable(my_force_test MyForceOptimizer.cpp)
target_link_libraries(my_force_test Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto motor_proto sensor_proto ${NLOPT_LIBRARIES})