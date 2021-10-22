#ifndef INPUT_DATA_HEADER
#define INPUT_DATA_HEADER

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
    ~Data();
    void print();
};
#endif