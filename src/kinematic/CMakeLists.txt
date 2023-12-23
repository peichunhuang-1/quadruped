add_library(LinkLegModel LinkLegModel.cpp)
add_library(LegModel Leg.cpp)
target_link_libraries(LegModel Eigen3::Eigen LinkLegModel)
