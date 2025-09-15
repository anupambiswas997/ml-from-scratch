#include "indexing_utils.hpp"
#include <algorithm>

ColumnSortFunctor::ColumnSortFunctor(const Matrix& X, size_t column)
{
    m_pX = &X;
    m_column = column;
}

bool ColumnSortFunctor::operator()(const size_t& i, const size_t& j)
{
    const std::vector<std::vector<double> > data = m_pX->getData();
    return data[i][m_column] < data[j][m_column];
}

std::vector<size_t> getColumnSortedIndices(const Matrix& X, size_t column, const std::vector<size_t>& indicesToInspect)
{
    std::vector<size_t> indices = indicesToInspect;
    std::sort(indices.begin(), indices.end(), ColumnSortFunctor(X, column));
    return indices;
}