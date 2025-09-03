#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"
#include "test_utils.hpp"

using namespace std;

void testLinearRegression(size_t sampleSize=1000, size_t numFeatures=1)
{
    Matrix X = getRandomMatrix(sampleSize, numFeatures, -3, 3);
    Vector weights = getRandomVector(numFeatures, -2, 2);
    double bias = getRandom();
    Vector noise = getRandomVector(sampleSize, -0.2, 0.2);
    Vector y = ((X * weights) + bias) + noise;

    LinearRegressionSolver linRegSolverAna = LinearRegressionSolver();
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - ANALYTICAL", linRegSolverAna.solve(X, y));
    Vector weightsAnalytical = linRegSolverAna.getWeights();
    double biasAnalytical = linRegSolverAna.getBias();

    double learningRate = 1.0e-4;
    size_t numStochasticSamples = 0; // as it is BGD and hence uses full batch
    size_t maxNumIterations = 100000;
    double tolerance = 1.0e-8;
    LinearRegressionGDSolver linRegSolverBGD = LinearRegressionGDSolver(sampleSize, numFeatures, learningRate, numStochasticSamples, maxNumIterations, tolerance);
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - BATCH GRADIENT DESCENT", linRegSolverBGD.solve(X, y));
    Vector weightsBGD = linRegSolverBGD.getWeights();
    double biasBGD = linRegSolverBGD.getBias();

    numStochasticSamples = size_t(0.4 * sampleSize);
    LinearRegressionGDSolver linRegSolverSGD = LinearRegressionGDSolver(sampleSize, numFeatures, learningRate, numStochasticSamples, maxNumIterations, tolerance);
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - STOCHASTIC GRADIENT DESCENT", linRegSolverSGD.solve(X, y));
    Vector weightsSGD = linRegSolverSGD.getWeights();
    double biasSGD = linRegSolverSGD.getBias();

    std::vector<std::string> headers = {"", "ACTUAL", "ANALYTICAL", "BATCH-GD", "STOCHASTIC-GD"};
    std::vector<std::vector<std::string> > data = {};
    data.push_back({"bias", std::to_string(bias), std::to_string(biasAnalytical), std::to_string(biasBGD), std::to_string(biasSGD)});

    for(size_t i = 0; i < numFeatures; i++)
    {
        std::string weightName = "weight-" + std::to_string(i);
        std::string wAct = std::to_string(weights[i]);
        std::string wAna = std::to_string(weightsAnalytical[i]);
        std::string wBGD = std::to_string(weightsBGD[i]);
        std::string wSGD = std::to_string(weightsSGD[i]);
        data.push_back({weightName, wAct, wAna, wBGD, wSGD});
    }
    std::cout << std::endl << getTableText(data, headers) << std::endl;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    testLinearRegression(1000, 5);
    return 0;
}