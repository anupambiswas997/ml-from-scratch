#include "test_utils.hpp"
#include "linear_regression_analytical_solver.hpp"
#include "linear_regression_GD_solver.hpp"
#include "logistic_regression_solver.hpp"
#include "decision_tree_regression_solver.hpp"
#include "kmeans_solver.hpp"
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"
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

void testDecisionTreeRegression(size_t sampleSize=1000, size_t numFeatures=1)
{
    // Prepare X and y data.
    // Data will be of the following pattern, like, for 2 dimensions (x1, x2):
    // y = a*x1*x1 + b*x2*x2 + c*x1 + d*x2 + e + noise
    vector<vector<double> > Xdata = {};
    vector<double> ydata = {};
    vector<double> constsAB = {};
    vector<double> constsCD = {};
    double constE = getRandom();
    for(size_t i = 0; i < sampleSize; i++)
    {
        vector<double> xrow = {};
        double noise = getRandom(-0.1, 0.1);
        double f = constE + noise;
        for(size_t j = 0; j < numFeatures; j++)
        {
            if(constsAB.size() == j)
            {
                constsAB.push_back(getRandom() / numFeatures);
            }
            if(constsCD.size() == j)
            {
                constsCD.push_back(getRandom() / numFeatures);
            }
            double xj = getRandom(-3, 3);
            f += (constsAB[j] * xj * xj + constsCD[j] * xj);
            xrow.push_back(xj);
        }
        Xdata.push_back(xrow);
        ydata.push_back(f);
    }

    Matrix X(Xdata);
    Vector y(ydata);
    DecisionTreeRegressionSolver DTSolver = DecisionTreeRegressionSolver(20, true);
    DTSolver.solve(X, y);
    DTSolver.describeTree();

    // Prepare X and y data for testing, separately.
    vector<vector<double> > XdataTest = {};
    vector<double> ydataTest = {};
    for(size_t i = 0; i < 100; i++)
    {
        vector<double> xrow = {};
        double f = constE;
        for(size_t j = 0; j < numFeatures; j++)
        {
            double xj = getRandom(-3, 3);
            xrow.push_back(xj);
            f += (constsAB[j] * xj * xj +  constsCD[j] * xj);
        }
        ydataTest.push_back(f);
        XdataTest.push_back(xrow);
    }
    Matrix XTest(XdataTest);
    Vector yTest(ydataTest);

    // Do the predictions using decision tree solver, for
    // both train and test data.
    Vector yPred = DTSolver.predict(X);
    Vector yTestPred = DTSolver.predict(XTest);

    // Write data in files for visualization purposes.
    writeXYData(X, y, "DTTrain.csv");
    writeXYData(X, yPred, "DTTrainPred.csv");
    writeXYData(XTest, yTest, "DTTest.csv");
    writeXYData(XTest, yTestPred, "DTTestPred.csv");

    // Compute mean square errors.
    cout << "Train MSE: " << getMeanSquareError(y, yPred) << endl;
    cout << "Test MSE : " << getMeanSquareError(yTest, yTestPred) << endl;
}

void testKMeansClustering(size_t sampleSize=1000, size_t numClusters=3)
{
    // number of features = 2, as that is best for visualization
    std::vector<std::vector<double> > Xdata = {};
    std::vector<Vector> centroids = {};
    double minDistBetweenCentroids = 3;
    double maxPointDistFromCentroid = 5;
    double minDistSq = minDistBetweenCentroids * minDistBetweenCentroids;
    // Determine spread-out centroids.
    // Centroids will be chosen in a way that they are far away from each other.
    for(size_t i = 0; i < numClusters; i++)
    {
        while(true)
        {
            bool centroidFound = true;
            Vector cen = getRandomVector(2, -10, 10);
            for(size_t j = 0; j < centroids.size(); j++)
            {
                Vector diff = cen - centroids[j];
                if(diff.dot(diff) < minDistSq)
                {
                    centroidFound = false;
                }
            }
            if(centroidFound)
            {
                centroids.push_back(cen);
            }
        }
    }
    double twoPI = 4 * atan(1);
    for(size_t i =0; i < sampleSize; i++)
    {
        // Randomly select a centroid to generate a point about.
        size_t centroidId = rand() % numClusters;
        double dist = getRandom(0, maxPointDistFromCentroid);
        double angle = getRandom(0, twoPI);
        const std::vector<double>& cen = centroids[centroidId].getData();
        double x = cen[0] + dist * cos(angle);
        double y = cen[1] + dist * sin(angle);
        Xdata.push_back({x, y});
    }
    Matrix X(Xdata);
    KMeansSolver kmeansSolver(3, 10);
    kmeansSolver.solve(X);
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    testLinearRegression(1000, 5);
    testLogisticRegression(1000, 5);
    testDecisionTreeRegression(1000);
    testKMeansClustering(1000, 3);
    return 0;
}