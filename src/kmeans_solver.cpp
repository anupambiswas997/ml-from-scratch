#include "kmeans_solver.hpp"
#include "matrix.hpp"
#include "index_shuffler.hpp"
#include <map>
#include <cassert>

KMeansSolver::KMeansSolver(size_t numCentroids, size_t numTrials)
:m_centroids({})
{
    m_numCentroids = numCentroids;
    m_numTrials = numTrials;
}

std::vector<Vector> getInitialCentroids(const Matrix& X, size_t numCentroids)
{
    // Currently, the centroids are being chosen randomly, but better strategies
    // are possible.
    std::vector<Vector> centroids = {};
    IndexShuffler indexer(X.getNumRows(), true);
    indexer.update();
    for(size_t i = 0; i < numCentroids; i++)
    {
        size_t index = indexer.getIndex(i);
        centroids.push_back(Vector(X.getData()[index]));
    }
    return centroids;
}

std::vector<size_t> getCentroidAssociations(const Matrix& X, const std::vector<Vector>& centroids)
{
    std::vector<size_t> centroidAssociations = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        size_t closestCentroidId;
        double closestCentroidDistSq;
        bool first = true;
        for(size_t j = 0; j < centroids.size(); j++)
        {
            Vector diff = Vector(X.getData()[i]) - centroids[j];
            double diffSq = diff.dot(diff);
            if(first || (diffSq < closestCentroidDistSq))
            {
                closestCentroidId = j;
                closestCentroidDistSq = diffSq;
                first = false;
            }
        }
        centroidAssociations.push_back(closestCentroidId);
    }
    return centroidAssociations;
}

std::pair<std::vector<Vector>, double> getUpdatedCentroids(const Matrix& X, const std::vector<Vector>& centroids)
{
    std::vector<Vector> updatedCentroids = {};
    double averageDistToCentroidsSqr = 0;
    std::map<size_t, Vector> centroidSums = {};
    std::map<size_t, size_t> counts = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        size_t closestCentroidId;
        double closestCentroidDistSq;
        bool first = true;
        for(size_t j = 0; j < centroids.size(); j++)
        {
            Vector diff = Vector(X.getData()[i]) - centroids[j];
            double diffSq = diff.dot(diff);
            if(first || (diffSq < closestCentroidDistSq))
            {
                closestCentroidDistSq = diffSq;
                closestCentroidId = j;
            }
            first = false;
        }
        if(centroidSums.count(closestCentroidId) == 0)
        {
            centroidSums[closestCentroidId] = Vector(std::vector<double>(X.getNumColumns(), 0));
            counts[closestCentroidId] = 0;
        }
        centroidSums[closestCentroidId] = centroidSums[closestCentroidId] + X.getData()[i];
        averageDistToCentroidsSqr += closestCentroidDistSq;
        counts[closestCentroidId]++;
    }
    averageDistToCentroidsSqr /= X.getNumRows();
    assert(centroidSums.size() == centroids.size());
    for(size_t i = 0; i < centroids.size(); i++)
    {
        updatedCentroids.push_back(centroidSums[i] * (1.0 / counts[i]));
    }
    return std::make_pair(updatedCentroids, averageDistToCentroidsSqr);
}

bool centroidsHaveChanged(const std::vector<Vector>& oldCentroids, const std::vector<Vector>& newCentroids, double tolerance=1.0e-8)
{
    assert(oldCentroids.size() == newCentroids.size());
    double tolSq = tolerance * tolerance;
    for(size_t i = 0; i < oldCentroids.size(); i++)
    {
        Vector diff = newCentroids[i] - oldCentroids[i];
        double diffSquared = diff.dot(diff);
        if(diffSquared > tolSq)
        {
            return true;
        }
    }
    return false;
}

std::pair<std::vector<Vector>, double> getConvergedCentroids(const Matrix& X, size_t numCentroids)
{
    std::vector<Vector> centroids = getInitialCentroids(X, numCentroids);
    double avgDistToCentroidSq = 0;
    bool shouldIterate = true;
    while(shouldIterate)
    {
        std::pair<std::vector<Vector>, double> updates = getUpdatedCentroids(X, centroids);
        shouldIterate = centroidsHaveChanged(centroids, updates.first);
        centroids = updates.first;
        avgDistToCentroidSq = updates.second;
    }
    return std::make_pair(centroids, avgDistToCentroidSq);
}

void KMeansSolver::solve(const Matrix& X)
{
    double avgDistToCentroidSqMin;
    bool first = true;
    for(size_t iter = 0; iter < m_numTrials; iter++)
    {
        std::pair<std::vector<Vector>, double> centroidsAndDistSqSum = getConvergedCentroids(X, m_numCentroids);
        if(first || (avgDistToCentroidSqMin > centroidsAndDistSqSum.second))
        {
            m_centroids = centroidsAndDistSqSum.first;
            avgDistToCentroidSqMin = centroidsAndDistSqSum.second;
        }
        first = false;
    }
}

const std::vector<Vector>& KMeansSolver::getCentroids() const
{
    return m_centroids;
}

const std::vector<double>& KMeansSolver::getSumOfSquares() const
{
    return m_distanceSquareSums;
}