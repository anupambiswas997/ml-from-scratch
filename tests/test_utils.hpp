#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>
#include <sstream>

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

typedef void (*NoParamsFuncP)();

// This function returns time taken by testFunc, in milliseconds.
double getTestTime(const NoParamsFuncP& testFunc)
{
    auto startTime = getMicroSeconds();
    testFunc();
    auto endTime = getMicroSeconds();
    return (endTime - startTime) / 1000.0;
}

std::string getPaddedString(std::string word, size_t maxLen)
{
    return (maxLen > word.length()) ? (std::string(maxLen - word.length(), ' ') + word) : word;
}

std::string getTableText(const std::vector<std::vector<std::string> >& data, const std::vector<std::string>& headers, size_t indent=0)
{
    std::string indentText = (indent > 0) ? std::string(indent, ' ') : "";
    std::vector<size_t> columnLengths = {};
    size_t numColumns = headers.size();
    for(size_t j = 0; j < numColumns; j++)
    {
        columnLengths.push_back(headers[j].length());
    }
    for(size_t i = 0; i < data.size(); i++)
    {
        assert(data[i].size() == numColumns);
        for(size_t j = 0; j < data[i].size(); j++)
        {
            size_t cellLength = data[i][j].length();
            columnLengths[j] = (cellLength > columnLengths[j]) ? cellLength : columnLengths[j];
        }
    }
    std::ostringstream ss;
    ss << indentText;
    size_t lastColumn = numColumns - 1;
    for(size_t j = 0; j < numColumns; j++)
    {
        std::string cellWord = " " + getPaddedString(headers[j], columnLengths[j]) + " ";
        ss << indentText << cellWord;
        if(j < lastColumn)
        {
            ss << "|";
        }
        else
        {
            ss << std::endl;
        }
    }
    for(size_t j = 0; j < numColumns; j++)
    {
        std::string cellWord = "-" + std::string(columnLengths[j], '-') + "-";
        ss << indentText << cellWord;
        if(j < lastColumn)
        {
            ss << "|";
        }
        else
        {
            ss << std::endl;
        }
    }
    for(size_t i = 0; i < data.size(); i++)
    {
        ss << indentText;
        for(size_t j = 0; j < numColumns; j++)
        {
            std::string cellWord = " " + getPaddedString(data[i][j], columnLengths[j]) + " ";
            ss << cellWord;
            if(j < lastColumn)
            {
                ss << "|";
            }
            else
            {
                ss << std::endl;
            }
        }
    }
    return ss.str();
}