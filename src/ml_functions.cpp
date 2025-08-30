#include "ml_functions.hpp"
#include <cmath>

double sigmoid(double z)
{
    double expMinusZ = exp(-z);
    return 1.0 / (1 + expMinusZ);
}

double sigmoid_d(double z)
{
    double sig = sigmoid(z);
    return sig * (1 - sig);
}