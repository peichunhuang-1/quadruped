find_package(NLopt REQUIRED)
add_executable(state_estimation estimation2.cpp)
target_link_libraries(state_estimation Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto LegModel LinkLegModel ${NLOPT_LIBRARIES})