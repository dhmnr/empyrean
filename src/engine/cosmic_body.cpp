#include "empyrean/engine/cosmic_body.hpp"

CosmicBody::CosmicBody(RealVector position, double mass, RealVector velocity = RealVector(0, 0, 0),
                       RealVector acceleration = RealVector(0, 0, 0))
    : position({position}), mass(mass), velocity(velocity), acceleration(acceleration) {}