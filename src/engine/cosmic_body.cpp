#include "empyrean/engine/cosmic_body.hpp"

CosmicBody::CosmicBody(RealVector position, double mass, RealVector velocity,
                       RealVector acceleration)
    : position({position}), mass(mass), velocity(velocity), acceleration(acceleration) {}