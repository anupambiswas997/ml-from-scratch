#include "linear_regression.hpp"
#include "linear_regression_GD_solver.hpp"
#include "logistic_regression_solver.hpp"
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"
#include "test_utils.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

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

    LinearRegressionAnalyticalSolver linRegSolverAna = LinearRegressionAnalyticalSolver();
    LinRegResult lrAna = getLinearRegressionTestResults(linRegSolverAna, X, y);

    double learningRate = 1.0e-4;
    size_t numStochasticSamples = 0; // as it is BGD and hence uses full batch
    size_t maxNumIterations = 100000;
    double tolerance = 1.0e-8;
    LinearRegressionGDSolver linRegSolverBGD = LinearRegressionGDSolver(learningRate, numStochasticSamples, maxNumIterations, tolerance);
    LinRegResult lrBGD = getLinearRegressionTestResults(linRegSolverBGD, X, y);

    numStochasticSamples = size_t(0.4 * sampleSize);
    LinearRegressionGDSolver linRegSolverSGD = LinearRegressionGDSolver(learningRate, numStochasticSamples, maxNumIterations, tolerance);
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

void getLogisticRegressionData(size_t sampleSize, size_t numFeatures, const Vector& planeNormal, double planeDistance, std::vector<bool>& yB, std::vector<std::vector<double> >& Xdata)
{
    size_t lastFeatureIndex = numFeatures - 1;
    for(size_t i = 0; i < sampleSize; i++)
    {
        double d = getRandom(-5, 5);
        yB.push_back((d > 0));
        std::vector<double> curData = {};
        double sum = 0;
        for(size_t j = 0; j < lastFeatureIndex; j++)
        {
            double val = getRandom(-5, 5);
            curData.push_back(val);
            sum += val * planeNormal[j];
        }
        double lastElement = (d + planeDistance - sum) / planeNormal[lastFeatureIndex];
        curData.push_back(lastElement);
        Xdata.push_back(curData);
    }
}

void testLogisticRegression(size_t sampleSize=1000, size_t numFeatures=1)
{
    Vector planePerp = getRandomVector(numFeatures, -3, 3);
    double planeDist = getRandom(0, 5);
    // test condition - last plane parameter shouldn't be too small in magnitude
    // this condition could be applied for any index arbitrarily
    // but for testing simplicity, being applied to the last index
    assert(fabs(planePerp[numFeatures - 1]) > 1.0e-6);

    std::vector<std::vector<double> > XTrainData = {};
    std::vector<bool> yBTrain = {};
    size_t lastFeatureIndex = numFeatures - 1;
    getLogisticRegressionData(sampleSize, numFeatures, planePerp, planeDist, yBTrain, XTrainData);
    Matrix XTrain = Matrix(XTrainData);

    double learningRate = 1.0e-4;
    size_t numStochasticSamples = size_t(0.5 * sampleSize);
    size_t maxNumIterations = 100000;
    double tolerance = 1.0e-8;
    LogisticRegressionSolver logRegSolver(learningRate, numStochasticSamples, maxNumIterations, tolerance);
    logRegSolver.solve(XTrain, yBTrain);

    std::vector<std::vector<double> > XTestData = {};
    std::vector<bool> yBTest = {};
    size_t testSize = 100;
    getLogisticRegressionData(testSize, numFeatures, planePerp, planeDist, yBTest, XTestData);
    Matrix XTest = Matrix(XTestData);
    std::vector<bool> yBPredict = logRegSolver.predictB(XTest);
    size_t numTruePositive = 0;
    size_t numTrueNegative = 0;
    size_t numFalsePositive = 0;
    size_t numFalseNegative = 0;
    for(size_t i = 0; i < testSize; i++)
    {
        if(yBTest[i] && yBPredict[i])
        {
            numTruePositive++;
        }
        else if(yBTest[i] && (!yBPredict[i]))
        {
            numFalseNegative++;
        }
        else if((!yBTest[i]) && yBPredict[i])
        {
            numFalsePositive++;
        }
        else if((!yBTest[i]) && (!yBPredict[i]))
        {
            numTrueNegative++;
        }
    }
    std::vector<std::vector<std::string> > data = {};
    std::vector<std::string> headers = {"", "TRUE-PREDICTED", "FALSE-PREDICTED"};
    data.push_back({"TRUE-ACTUAL", std::to_string(numTruePositive), std::to_string(numFalseNegative)});
    data.push_back({"FALSE-ACTUAL", std::to_string(numFalsePositive), std::to_string(numTrueNegative)});
    std::cout << std::endl << "Logistic regression test" << std::endl << "CONFUSION MATRIX:" << std::endl;
    std::cout << getTableText(data, headers) << std::endl;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    testLinearRegression(1000, 5);
    testLogisticRegression(1000, 5);
    return 0;
}