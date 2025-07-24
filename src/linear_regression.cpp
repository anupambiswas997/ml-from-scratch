#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "sparse_matrix.hpp"

LinearRegressionSolver::LinearRegressionSolver()
{
    std::cout << "Created LinearRegression object" << std::endl;
}

void LinearRegressionSolver::solve(const Matrix& X, const Vector& y)
{
    size_t m = X.getNumRows();
    double oneByM = 1.0 / m;
    Matrix XT = X.getTranspose();//getTranspose(X);
    SparseMatrix iden(0, m, m);// = getIdentityMatrix(m);
    for(size_t i = 0; i < m; i++)
    {
        iden[i][i] = 1;
    }
    SparseMatrix U = iden - oneByM;
    /*
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
        {
            U[i][j] -= oneByM;
        }
    }//*/
    Matrix XTU = XT * U;//getProductOfMatrices(XT, U);
    Matrix XTUX = XTU * X;
    Matrix XTUXinverse = XTUX.getInverse();//getMatrixInverse(getProductOfMatrices(XTU, X));
    Vector XTUy = XTU * y;//getMatrixVectorProduct(XTU, y);
    m_weights = XTUXinverse * XTUy;//getMatrixVectorProduct(XTUXinverse, XTUy);
    double sum = 0;
    Vector Xw = X * m_weights;//getMatrixVectorProduct(X, m_weights);
    for(size_t i = 0; i < m; i++)
    {
        sum += (y[i] - Xw[i]);
    }
    m_bias = oneByM * sum;
}