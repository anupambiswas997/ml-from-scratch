#ifndef INDEX_SHUFFLER_HPP
#define INDEX_SHUFFLER_HPP

#include <vector>

class IndexShuffler
{
    std::vector<size_t> m_indices;
    bool m_doShuffle;
public:
    IndexShuffler();
    IndexShuffler(size_t size, bool doShuffle);
    void update();
    size_t getIndex(size_t i);
};

#endif