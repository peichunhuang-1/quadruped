# add_library(EKF EKF.cpp)
# target_link_libraries(EKF Eigen3::Eigen LegModel LinkLegModel)
# add_executable(state_ekf test_ekf.cpp)
# target_link_libraries(state_ekf Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto LegModel LinkLegModel EKF motor_proto)


add_library(KF KF.cpp)
target_link_libraries(KF Eigen3::Eigen LegModel LinkLegModel)
add_executable(state_kf test_kf.cpp)
target_link_libraries(state_kf Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto LegModel LinkLegModel KF motor_proto)

# add_executable(state_kf_webot KF_webot.cpp)
# target_link_libraries(state_kf_webot Eigen3::Eigen ${_CORE_LIBRARIES} geometry_proto sensor_proto LegModel LinkLegModel KF motor_proto robot_proto)
# target_compile_definitions(state_kf_webot PRIVATE -DSIMULATION)