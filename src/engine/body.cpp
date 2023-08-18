#include "empyrean/engine/body.hpp"

Body::Body(glm::dvec3 position, double mass, glm::dvec3 velocity)
    : position(position), mass(mass), velocity(velocity) {}
