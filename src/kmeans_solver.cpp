#include "kmeans_solver.hpp"
#include "matrix.hpp"

KMeansSolver::KMeansSolver(size_t numCentroids, size_t numTrials)
{
    m_numCentroids = numCentroids;
    m_numTrials = numTrials;
}

void KMeansSolver::solve(const Matrix& X)
{
}

const std::vector<Vector>& KMeansSolver::getCentroids() const
{
    return m_centroids;
}

const std::vector<double>& KMeansSolver::getSumOfSquares() const
{
    return m_distanceSquareSums;
}