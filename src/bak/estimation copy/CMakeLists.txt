find_package(NLopt REQUIRED)
add_library(ObservationModel ObservationModel.cpp)
target_link_libraries(ObservationModel Eigen3::Eigen LegModel LinkLegModel)
add_executable(state_estimation estimation.cpp)
target_link_libraries(state_estimation Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto LegModel LinkLegModel ${NLOPT_LIBRARIES} ObservationModel motor_proto)

add_executable(record_webot record_data.cpp)
target_link_libraries(record_webot Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto motor_proto LegModel LinkLegModel ${NLOPT_LIBRARIES} ObservationModel)
target_compile_definitions(record_webot PRIVATE -DSIMULATION)