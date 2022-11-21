#ifndef DATA_HEADER
#define DATA_HEADER

#include "MemoryHandler.hpp"

#include <string>
#include <iostream>
#include <new>

class Data
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
    // Total Number of Elements
    unsigned lengthElements;

    // Vector of Names for each Element
    Pointer<std::string> names;

    // Boolean value to represent the Data was properly loaded to the Pointer
    bool isValid;
    // Count of Data Entries added to the Data Class
    unsigned count;

    // Index Value for Multiple Data Type
    unsigned multipleIndex;
    // Length of each Multiple Data Type
    unsigned multipleLenght;

    // Delete Pointer function
    void DeletePointer();

public:
    // Default Constructor
    Data();
    // Constructor with Maximum Number of Elements
    Data(unsigned lengthElements_in);
    // Constructor to copy an instance of Data class
    Data(const Data &data_in);
    // Add Data Entry to Data Class
    unsigned Add(std::string name_in, unsigned length_in);
    // Add Multiple Data Entries to Data Class
    void Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in);
    // Initialize Data Pointers
    void Initialize(PointerType type_in, PointerContext context_in);
    // Instantiate Contiguous Pointers for multiple copied Data Intances
    static void InstantiateMultiple(Pointer<Data> &dataArray_out, const Data &data_in, unsigned length_in, bool resetValues = false);
    // Load data by index
    void LoadData(unsigned index_in, Pointer<double> array_in, unsigned length_in);
    // Load multiple data by index
    void LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<double>> array_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in);
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
    // Get Length of a given Data Entry
    unsigned GetLength(unsigned index_in) const;
    // Get Name of a given Data Entry
    std::string GetNames(unsigned index_in) const;
    // Access Pointer of a given Data Entry
    Pointer<double> &operator[](unsigned index);
    // Default Destructor
    ~Data();
};

#endif