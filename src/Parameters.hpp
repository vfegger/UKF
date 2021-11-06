#ifndef PARAMETERS_HEADER
#define PARAMETERS_HEADER

#include <string>

class Parameters
{
private:
    std::string name;
    int* parameters;
    unsigned length;
public:
    Parameters();
    Parameters(std::string name, double* data, unsigned length);
    ~Parameters();
    unsigned GetLength();
    std::string GetName();
    int& operator[](unsigned index);
    const int& operator[](unsigned index) const;
    void print();
};
#endif