#include "linear_regression_analytical_solver.hpp"
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"

LinearRegressionAnalyticalSolver::LinearRegressionAnalyticalSolver()
{
}

void LinearRegressionAnalyticalSolver::solve(const Matrix& X, const Vector& y)
{
    size_t m = X.getNumRows();
    double oneByM = 1.0 / m;
    Matrix XT = X.getTranspose();
    SparseMatrix iden(0, m, m);
    for(size_t i = 0; i < m; i++)
    {
        iden[i][i] = 1;
    }
    SparseMatrix U = iden - oneByM;
    Matrix XTU = XT * U;
    Matrix XTUX = XTU * X;
    Matrix XTUXinverse = XTUX.getInverse();
    Vector XTUy = XTU * y;
    m_weights = XTUXinverse * XTUy;
    double sum = 0;
    Vector Xw = X * m_weights;
    for(size_t i = 0; i < m; i++)
    {
        sum += (y[i] - Xw[i]);
    }
    m_bias = oneByM * sum;
}