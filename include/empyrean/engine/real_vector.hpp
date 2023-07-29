#pragma once

#include <iostream>

class RealVector {
public:
  double x;
  double y;
  double z;
  double magnitude;
  RealVector(double x = 0, double y = 0, double z = 0);
  double GetMagnitude();
  RealVector GetUnitVector();
  RealVector operator+(RealVector& rv);
  RealVector operator*(double val);
  RealVector& operator+=(const RealVector& rv);
  friend std::ostream& operator<<(std::ostream& os, const RealVector& vec);
};

RealVector GetDistance(const RealVector& v1, const RealVector& v2);