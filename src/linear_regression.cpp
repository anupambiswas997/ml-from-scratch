#include "linear_regression.hpp"
#include <iostream>
#include "linear_algebra.hpp"

LinearRegressionSolver::LinearRegressionSolver()
{
    std::cout << "Created LinearRegression object" << std::endl;
}

void LinearRegressionSolver::solve(const Matrix& X, const Vector& y)
{
    //int h = h;std::cout << "h is " << h << std::endl;return;
    int m = X.size();
    double oneByM = 1.0 / m;
    Matrix XT = getTranspose(X);
    Matrix iden = getIdentityMatrix(m);
    Matrix U = iden;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
        {
            U[i][j] -= oneByM;
        }
    }
    Matrix XTU = getProductOfMatrices(XT, U);
    Matrix XTUXinverse = getMatrixInverse(getProductOfMatrices(XTU, X));
    Vector XTUy = getMatrixVectorProduct(XTU, y);
    m_weights = getMatrixVectorProduct(XTUXinverse, XTUy);
    double sum = 0;
    Vector Xw = getMatrixVectorProduct(X, m_weights);
    for(int i = 0; i < m; i++)
    {
        sum += (y[i] - Xw[i]);
    }
    m_bias = oneByM * sum;
}