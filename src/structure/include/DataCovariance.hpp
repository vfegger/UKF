#ifndef DATA_COVARIANCE_HEADER
#define DATA_COVARIANCE_HEADER

#include <string>
#include <iostream>
#include <new>

#include "Data.hpp"

enum DataCovarianceMode{
    Natural,Compact,Complete
};

class DataCovariance
{
private:
    double* pointer;
    unsigned length;

    double** offsetPointer;
    unsigned* lengthArray;
    unsigned* offsetArray;
    unsigned lengthElements;

    std::string* names;

    bool isValid;
    unsigned count;
    
    void DeletePointer();
public: 
    DataCovariance();
    DataCovariance(unsigned lengthElements_in);
    DataCovariance(const Data& data_in);
    DataCovariance(const DataCovariance& dataCovariance_in);
    unsigned Add(std::string name_in, unsigned length_in);
    void Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    void Initialize();
    void LoadData(unsigned index_in, double* array_in, unsigned length_in, DataCovarianceMode mode_in);
    void LoadData(unsigned* indexes_in, double** array_in, unsigned* lengthArray_in, DataCovarianceMode* modeArray_in, unsigned lengthElements_in);
    unsigned GetCapacity() const;
    double* GetPointer() const;
    bool GetValidation() const;
    unsigned GetCount() const;
    unsigned GetLength() const;
    unsigned GetLength(unsigned index_in) const;
    unsigned GetOffset(unsigned index_in) const;
    std::string GetNames(unsigned index_in) const;
    double*& operator[](unsigned index);
    ~DataCovariance();
};

#endif