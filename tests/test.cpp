#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"
#include "test_utils.hpp"

using namespace std;

typedef struct LinearRegressionTestResult
{
    double bias;
    Vector weights;
    double timeElapsed;
    LinearRegressionTestResult(double _b, const Vector& _w, double _t)
    {
        bias = _b;
        weights = _w;
        timeElapsed = _t;
    }
} LinRegResult;

template<typename T>
LinRegResult getLinearRegressionTestResults(T& solver, const Matrix& X, const Vector& y)
{
    auto tStart = getMicroSeconds();
    solver.solve(X, y);
    auto tEnd = getMicroSeconds();
    return LinRegResult(solver.getBias(), solver.getWeights(), (tEnd - tStart) / 1000.0);
}

void testLinearRegression(size_t sampleSize=1000, size_t numFeatures=1)
{
    Matrix X = getRandomMatrix(sampleSize, numFeatures, -3, 3);
    Vector weights = getRandomVector(numFeatures, -2, 2);
    double bias = getRandom();
    Vector noise = getRandomVector(sampleSize, -0.2, 0.2);
    Vector y = ((X * weights) + bias) + noise;

    LinearRegressionSolver linRegSolverAna = LinearRegressionSolver();
    LinRegResult lrAna = getLinearRegressionTestResults(linRegSolverAna, X, y);

    double learningRate = 1.0e-4;
    size_t numStochasticSamples = 0; // as it is BGD and hence uses full batch
    size_t maxNumIterations = 100000;
    double tolerance = 1.0e-8;
    LinearRegressionGDSolver linRegSolverBGD = LinearRegressionGDSolver(sampleSize, numFeatures, learningRate, numStochasticSamples, maxNumIterations, tolerance);
    LinRegResult lrBGD = getLinearRegressionTestResults(linRegSolverBGD, X, y);

    numStochasticSamples = size_t(0.4 * sampleSize);
    LinearRegressionGDSolver linRegSolverSGD = LinearRegressionGDSolver(sampleSize, numFeatures, learningRate, numStochasticSamples, maxNumIterations, tolerance);
    LinRegResult lrSGD = getLinearRegressionTestResults(linRegSolverSGD, X, y);

    std::vector<std::string> headers = {"", "ACTUAL", "ANALYTICAL", "BATCH-GD", "STOCHASTIC-GD"};
    std::vector<std::vector<std::string> > data = {};
    data.push_back({"bias", std::to_string(bias), std::to_string(lrAna.bias), std::to_string(lrBGD.bias), std::to_string(lrSGD.bias)});

    for(size_t i = 0; i < numFeatures; i++)
    {
        std::string weightName = "weight-" + std::to_string(i);
        std::string wAct = std::to_string(weights[i]);
        std::string wAna = std::to_string(lrAna.weights[i]);
        std::string wBGD = std::to_string(lrBGD.weights[i]);
        std::string wSGD = std::to_string(lrSGD.weights[i]);
        data.push_back({weightName, wAct, wAna, wBGD, wSGD});
    }
    data.push_back({"time (ms)", "N/A", std::to_string(lrAna.timeElapsed), std::to_string(lrBGD.timeElapsed), std::to_string(lrSGD.timeElapsed)});
    std::cout << std::endl << getTableText(data, headers) << std::endl;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    testLinearRegression(1000, 5);
    return 0;
}