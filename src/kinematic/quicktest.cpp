#include "Leg.hpp"

int main() {
    Leg leg(Eigen::Vector3d(0, 0, 0));
    leg.Calculate(160.0 / 180.0 * M_PI, 0, 0, 0, 0, 0);
    std::cout << std::abs(leg.G) << "\n";
}