find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(image_transfer image.cpp)
target_link_libraries(image_transfer Eigen3::Eigen ${_CORE_LIBRARIES} doc_proto ${OpenCV_LIBS})

add_executable(image_listen image_listen.cpp)
target_link_libraries(image_listen Eigen3::Eigen ${_CORE_LIBRARIES} doc_proto ${OpenCV_LIBS})