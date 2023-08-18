#pragma once

#include <glm/glm.hpp>

class Body {
public:
  glm::dvec3 position, velocity, acceleration;
  double mass;
  Body(glm::dvec3 position, double mass, glm::dvec3 velocity = {0.0, 0.0, 0.0});
};
