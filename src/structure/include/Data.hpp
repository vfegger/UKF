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
    unsigned count;
    
    unsigned multipleIndex;
    unsigned multipleLenght;

    void DeletePointer();
public:
    Data();
    Data(unsigned lengthElements_in);
    Data(const Data& data_in);
    unsigned Add(std::string name_in, unsigned length_in);
    void Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    void Initialize();
    static void InstantiateMultiple(Data*& dataArray_out, const Data& data_in, unsigned length_in, bool resetValues = false);
    void LoadData(unsigned index_in, double* array_in, unsigned length_in);
    void LoadData(unsigned* indexes_in, double** array_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    unsigned GetCapacity() const;
    double* GetPointer() const;
    bool GetValidation() const;
    unsigned GetCount() const;
    unsigned GetLength() const;
    unsigned GetLength(unsigned index_in) const;
    std::string GetNames(unsigned index_in) const;
    double*& operator[](unsigned index);
    ~Data();
};

#endif