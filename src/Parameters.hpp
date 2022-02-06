#ifndef PARAMETERS_HEADER
#define PARAMETERS_HEADER

#include <string>

class Parameters_Int
{
private:
    std::string name;
    long long int* parameters;
    unsigned length;
public:
    Parameters_Int();
    Parameters_Int(std::string name_input, long long int* parameters_input, unsigned length_input);
    Parameters_Int(const Parameters_Int& parameters_input);
    ~Parameters_Int();
    Parameters_Int& operator=(const Parameters_Int& rhs);
    unsigned GetLength();
    std::string GetName();
    long long int& operator[](unsigned index);
    const long long int& operator[](unsigned index) const;
    void print();
};

class Parameters_UInt
{
private:
    std::string name;
    long long unsigned* parameters;
    unsigned length;
public:
    Parameters_UInt();
    Parameters_UInt(std::string name_input, long long unsigned* parameters_input, unsigned length_input);
    Parameters_UInt(const Parameters_UInt& parameters_input);
    ~Parameters_UInt();
    Parameters_UInt& operator=(const Parameters_UInt& rhs);
    unsigned GetLength();
    std::string GetName();
    long long unsigned& operator[](unsigned index);
    const long long unsigned& operator[](unsigned index) const;
    void print();
};

class Parameters_FP
{
private:
    std::string name;
    double* parameters;
    unsigned length;
public:
    Parameters_FP();
    Parameters_FP(std::string name_input, double* parameters_input, unsigned length_input);
    Parameters_FP(const Parameters_FP& parameters_input);
    ~Parameters_FP();
    Parameters_FP& operator=(const Parameters_FP& rhs);
    unsigned GetLength();
    std::string GetName();
    double& operator[](unsigned index);
    const double& operator[](unsigned index) const;
    void print();
};

class Parameters{
private:
    unsigned Int_Length;
    unsigned UInt_Length;
    unsigned FP_Length;

public:
    Parameters_Int* Int;
    Parameters_UInt* UInt;
    Parameters_FP* FP;

    Parameters();
    Parameters(Parameters_Int* pointer_int, unsigned int_length, Parameters_UInt* pointer_uint, unsigned uint_length, Parameters_FP* pointer_fp, unsigned fp_length);
    ~Parameters();
};

#endif