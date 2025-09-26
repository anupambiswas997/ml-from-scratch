#include "kmeans_solver.hpp"
#include "matrix.hpp"
#include "index_shuffler.hpp"

KMeansSolver::KMeansSolver(size_t numCentroids, size_t numTrials)
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

std::vector<Vector> getConvergedCentroids(const Matrix& X, const std::vector<Vector>& initialCentroids)
{
    std::vector<Vector> centroids = initialCentroids;
    std::vector<size_t> oldCentroidAssociations = {};
    bool cond = true;
    bool firstIter = true;
    while(cond)
    {
        std::vector<size_t> newCentroidAssociations = {};
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
            newCentroidAssociations.push_back(closestCentrodId);
        }
        if(!firstIter)
        {}
        cond = false;
        firstIter = false;
    }
    return centroids;
}

void KMeansSolver::solve(const Matrix& X)
{
    for(size_t iter = 0; iter < m_numTrials; iter++)
    {
        std::vector<Vector> initialCentroids = getInitialCentroids(X, m_numCentroids);
        std::vector<Vector> convergedCentroids = getConvergedCentroids(X, initialCentroids);
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