add_executable(cpg_node cpg.cpp cpg.hpp)
target_link_libraries(cpg_node Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto)

add_executable(cpg_node_webot cpg.cpp cpg.hpp)
target_link_libraries(cpg_node_webot Eigen3::Eigen ${_CORE_LIBRARIES} robot_proto geometry_proto)
target_compile_definitions(cpg_node_webot PRIVATE -DSIMULATION)