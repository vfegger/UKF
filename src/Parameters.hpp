#ifndef PARAMETERS_HEADER
#define PARAMETERS_HEADER

#include <string>

class Parameters
{
protected:
    std::string name;
    void* parameters;
    unsigned length;
    unsigned sizeType;
public:
    Parameters();
    Parameters(std::string name_input, void* parameters_input, unsigned length_input, unsigned sizeType_input);
    Parameters(Parameters& parameters_input);
    ~Parameters();
    Parameters& operator=(const Parameters& rhs);
    unsigned GetLength();
    std::string GetName();
    template<class T> void SetValue(T& value, unsigned index);
    template<class T> const T& GetValue(unsigned index) const;
    template<class T> void print();
};

#endif