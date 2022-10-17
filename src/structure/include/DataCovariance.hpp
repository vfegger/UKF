#ifndef DATA_COVARIANCE_HEADER
#define DATA_COVARIANCE_HEADER

#include <string>
#include <iostream>
#include <new>

#include "Data.hpp"
#include "MemoryHandler.hpp"

enum DataCovarianceMode{
    Natural,Compact,Complete
};

class DataCovariance
{
private:
    // Main Pointer
    Pointer<double> pointer;
    // Total Length Allocated
    unsigned length;

    // Vector of pointers for each Element
    Pointer<Pointer<double>> offset;
    // Vector of Lengths for each Element
    Pointer<unsigned> lengthOffset;
    // Vector of Offsets for each Element
    Pointer<unsigned> offsetArray;
    // Total Number of Elements
    unsigned lengthElements;

    // Vector of Names for each Element
    Pointer<std::string> names;

    // Boolean value to represent the Data was properly loaded to the Pointer
    bool isValid;
    // Count of Data Entries added to the Data Class
    unsigned count;
    
    // Delete Pointer function
    void DeletePointer();
public: 
    // Default Constructor
    DataCovariance();
    // Constructor with Maximum Number of Elements
    DataCovariance(unsigned lengthElements_in);
    // Constructor to link an instance of Data Covariance class to a Data class
    DataCovariance(const Data& data_in);
    // Constructor to copy an instance of Data Covariance class
    DataCovariance(const DataCovariance& dataCovariance_in);
    // Add Data Covariance Entry to Data Covariance Class
    unsigned Add(std::string name_in, unsigned length_in);
    // Add Multiple Data Covariance Entries to Data Covariance Class
    void Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in);
    // Initialize Data Pointers
    void Initialize();
    // Load data by index
    void LoadData(unsigned index_in, Pointer<double> array_in, unsigned length_in, DataCovarianceMode mode_in);
    // Load multiple data by index
    void LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<double>> array_in, Pointer<unsigned> lengthArray_in, Pointer<DataCovarianceMode> modeArray_in, unsigned lengthElements_in);
    // Get Maximum Capacity
    unsigned GetCapacity() const;
    // Get Main Pointer
    Pointer<double> GetPointer() const;
    // Get Boolean Value that represents if the Main Pointer is valid for use
    bool GetValidation() const;
    // Get Current Count of Entries added
    unsigned GetCount() const;
    // Get Length of the Main Pointer
    unsigned GetLength() const;
    // Get Length of a given Data Covariance Entry
    unsigned GetLength(unsigned index_in) const;
    // Get Offset Index of a given Data Covariance Entry
    unsigned GetOffset(unsigned index_in) const;
    // Get Name of a given Data Covariance Entry
    std::string GetNames(unsigned index_in) const;
    // Access Pointer of a given Data Covariance Entry
    Pointer<double> operator[](unsigned index);
    // Default Destructor
    ~DataCovariance();
};

#endif