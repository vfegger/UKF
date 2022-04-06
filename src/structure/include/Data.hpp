#ifndef DATA_HEADER
#define DATA_HEADER

#include <string>
#include <iostream>
#include <new>

class Data {
private:
    double* pointer;
    unsigned length;

    double** offsetPointer;
    unsigned* lengthArray;
    unsigned lengthElements;

    std::string* names;

    bool isValid;
    bool isMultiple;
    unsigned indexMultiple;
    unsigned count;

    void DeletePointer();
public:
    Data(unsigned lengthElements_in);
    Data(const Data& data_in);
    unsigned Add(std::string name_in, unsigned length_in);
    void Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    void Initialize();
    static void InitializeMultiple(Data* dataArray_in, unsigned length_in);
    void LoadData(unsigned index_in, double* array_in, unsigned length_in);
    void LoadData(unsigned* indexes_in, double** array_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    double* GetPointer();
    unsigned GetLength();
    bool GetValidation();
    double*& operator[](unsigned index);
    static void DeleteMultiple(Data* dataArray_in, unsigned length_in);
    ~Data();
};

#endif