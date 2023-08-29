#pragma once

const double GRAVITY_CONSTANT = 6.6743e-11;

struct InitialState {
  std::vector<Body> bodies = {};
  size_t objCount;
  double gravityConstant = GRAVITY_CONSTANT;
  double timeStep;
  double scaleFactor;
};