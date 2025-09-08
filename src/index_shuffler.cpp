#include "index_shuffler.hpp"
#include <cstdlib>

IndexShuffler::IndexShuffler()
{
    m_indices = {};
    m_doShuffle = false;
}

IndexShuffler::IndexShuffler(size_t size, bool doShuffle)
{
    m_indices = {};
    for(size_t i = 0; i < size; i++)
    {
        m_indices.push_back(i);
    }
    m_doShuffle = doShuffle;
}

void IndexShuffler::update()
{
    if(!m_doShuffle)
    {
        return;
    }
    for(size_t i = 0; i < m_indices.size(); i++)
    {
        size_t i2 = rand() % m_indices.size();
        size_t temp = m_indices[i];
        m_indices[i] = m_indices[i2];
        m_indices[i2] = temp;
    }
}

size_t IndexShuffler::getIndex(size_t i)
{
    return m_indices[i];
}