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
    Parameters(std::string name, int* data, unsigned length);
    Parameters(Parameters& parameters_input);
    ~Parameters();
    Parameters& operator=(const Parameters& rhs);
    unsigned GetLength();
    std::string GetName();
    int& operator[](unsigned index);
    const int& operator[](unsigned index) const;
    void print();
};
#endif