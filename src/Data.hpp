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
    Data(Data& data_input);
    ~Data();
    unsigned GetLength();
    double& operator[](unsigned index);
    const double& operator[](unsigned index) const;
    void print();
};
#endif