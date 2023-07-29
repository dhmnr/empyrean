#include "empyrean/engine/cosmic_body.hpp"

CosmicBody::CosmicBody(RealVector position, double mass, RealVector velocity,
                       RealVector initialAcceleration)
    : position({position}),
      mass(mass),
      velocity(velocity),
      initialAcceleration(initialAcceleration) {}