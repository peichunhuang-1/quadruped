add_executable(imu_cx5 imu.cpp)
target_link_libraries(imu_cx5 Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto  sensor_proto mip "${SERIAL_LIB}" )
target_compile_definitions(imu_cx5 PUBLIC "MIP_USE_SERIAL")

add_executable(imu_listen imu_listen.cpp)
target_link_libraries(imu_listen geometry_proto  sensor_proto ${_CORE_LIBRARIES})