#ifndef KMEANS_SOLVER_HPP
#define KMEANS_SOLVER_HPP

#include "vectr.hpp"
#include <vector>

class Matrix;

class KMeansSolver
{
    std::vector<Vector> m_centroids;
    std::vector<double> m_distanceSquareSums;
    size_t m_numCentroids;
    size_t m_numTrials;
public:
    KMeansSolver(size_t numCentroids, size_t numTrials);
    void solve(const Matrix& X);
    const std::vector<Vector>& getCentroids() const;
    const std::vector<double>& getSumOfSquares() const;
};

#endif