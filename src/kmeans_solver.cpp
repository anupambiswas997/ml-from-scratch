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

std::vector<size_t> getCentroidAssociations(const Matrix& X, std::vector<Vector>& centroids)
{
    std::vector<size_t> centroidAssociations = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        size_t closestCentrodId;
        double closestCentroidDistSq;
        bool first = true;
        for(size_t j = 0; j < centroids.size(); j++)
        {
            Vector diff = Vector(X.getData()[i]) - centroids[j];
            double diffSq = diff.dot(diff);
            if(first || (diffSq < closestCentroidDistSq))
            {
                closestCentroidDistSq = diffSq;
                closestCentrodId = j;
            }
            first = false;
        }
        centroidAssociations.push_back(closestCentrodId);
    }
    return centroidAssociations;
}

std::vector<Vector> getUpdatedCentroids(const Matrix& X, const std::vector<size_t>& centroidAssociations)
{
    std::vector<Vector> centroids = {};
    std::map<size_t, Vector> centroidSums = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        size_t assoc = centroidAssociations[i];
        if(centroidSums.count(assoc) == 0)
        {
            centroidSums[assoc] = Vector(std::vector<double>(X.getNumColumns(), 0.0));
        }
        centroidSums[assoc] = centroidSums[assoc] + Vector(X.getData()[i]);
    }
    for(size_t i = 0; i < centroidSums.size(); i++)
    {
        centroids.push_back(centroidSums[i] * (1.0 / centroidSums[i].size()));
    }
    return centroids;
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

std::pair<std::vector<Vector>, double> getConvergedCentroids(const Matrix& X, const std::vector<Vector>& initialCentroids)
{
    std::vector<Vector> centroids = initialCentroids;
    double avgDistToCentroidSq = 0;
    bool shouldIterate = true;
    while(shouldIterate)
    {
        std::vector<size_t> centroidAssociations = getCentroidAssociations(X, centroids);
        std::vector<Vector> updatedCentroids = getUpdatedCentroids(X, centroidAssociations);
        shouldIterate = centroidsHaveChanged(centroids, updatedCentroids);
        centroids = updatedCentroids;
        if(!shouldIterate)
        {
            for(size_t i = 0; i < X.getNumRows(); i++)
            {
                Vector diff = Vector(X.getData()[i]) - centroids[centroidAssociations[i]];
                avgDistToCentroidSq += diff.dot(diff);
            }
        }
    }
    avgDistToCentroidSq /= X.getNumRows();
    return std::make_pair(centroids, avgDistToCentroidSq);
}

void KMeansSolver::solve(const Matrix& X)
{
    double avgDistToCentroidSqMin;
    bool first = true;
    for(size_t iter = 0; iter < m_numTrials; iter++)
    {
        std::vector<Vector> initialCentroids = getInitialCentroids(X, m_numCentroids);
        std::pair<std::vector<Vector>, double> centroidsAndDistSqSum = getConvergedCentroids(X, initialCentroids);
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