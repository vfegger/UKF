#ifndef DATA_HEADER
#define DATA_HEADER

#include <string>

class Data
{
private:
    std::string name;
    double* data;
    unsigned length;
public:
    Data();
    Data(std::string name, double* data, unsigned length);
    Data(const Data& data_input);
    ~Data();
    Data& operator=(const Data& rhs);
    unsigned GetLength();
    std::string GetName();
    double& operator[](unsigned index);
    const double& operator[](unsigned index) const;
    void print();
};
#endif