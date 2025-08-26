#include <iostream>
#include <chrono>

#define SHOW_TIME_ELAPSED(MSG , X) \
{ \
auto tStart = getMicroSeconds(); \
X; \
auto tEnd = getMicroSeconds(); \
auto tDiff = tEnd - tStart; \
std::cout << MSG << std::endl; \
std::cout << "Time elapsed: " << double(tDiff) / 1000.0 << " milliseconds" << std::endl; \
}

long long getMicroSeconds()
{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}