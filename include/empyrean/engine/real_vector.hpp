class RealVector {
public:
  double x;
  double y;
  double z;
  double magnitude;
  RealVector(double x, double y, double z);
  RealVector GetUnitVector();
  RealVector operator+(RealVector& rv);
  RealVector operator*(double val);
};