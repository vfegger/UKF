#ifndef PARAMETER_HEADER
#define PARAMETER_HEADER

#include "MemoryHandler.hpp"

#include <string>
#include <iostream>
#include <new>

class Parameter
{
private:
    // Main Pointer
    Pointer<void> pointer;
    // Total Length Allocated
    unsigned length;

    // Vector of pointers for each Element
    Pointer<Pointer<void>> offset;
    // Vector of Type's sizes for each Element
    Pointer<unsigned> sizeTypeArray;
    // Vector of Lengths for each Element
    Pointer<unsigned> lengthArray;
    // Total Number of Elements
    unsigned lengthElements;

    // Vector of Names for each Element
    Pointer<std::string> names;

    // Boolean value to represent the Parameters was properly loaded to the Pointer
    bool isValid;
    // Count of Parameter Entries added to the Parameter Class
    unsigned count;

public:
    // Default Constructor
    Parameter();
    // Constructor with Maximum Numer of Elements
    Parameter(unsigned lengthElements_in);
    // Constructor to copy an instance of Parameter class
    Parameter(const Parameter &parameter_in);
    // Copy assignment of an instance of Parameter class
    Parameter& operator=(const Parameter &parameter_in);
    // Add Parameter Entry to Parameter Class
    unsigned Add(std::string name_in, unsigned length_in, unsigned sizeType_in);
    // Add Multiple Parameter Entries to Data Class
    void Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, Pointer<unsigned> sizeTypeArray_in, unsigned lengthElements_in);
    // Initialize Parameter Pointers
    void Initialize(PointerType type_in, PointerContext context_in);
    // Load data by index
    template <typename T>
    void LoadData(unsigned index_in, Pointer<T> array_in, unsigned length_in);
    // Load multiple data by index
    template <typename T>
    void LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<T>> array_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in);
    // Get Parameter Entry Pointer
    template <typename T>
    Pointer<T> GetPointer(unsigned index);
    // Get Length of a given Parameter Entry
    unsigned GetLength(unsigned index);
    // Get Boolean Value that represents if the Main Pointer is valid for use
    bool GetValidation();
    // Default Destructor
    ~Parameter();
};

template <typename T>
void Parameter::LoadData(unsigned index_in, Pointer<T> array_in, unsigned length_in)
{
    if (isValid == false)
    {
        std::cout << "Error: Load while structure is not initialized.";
        return;
    }
    if (index_in >= count)
    {
        std::cout << "Error: Index is out of range.";
        return;
    }
    MemoryHandler::Copy<T>(Pointer<T>((T*)(offset.pointer[index_in].pointer),offset.pointer[index_in].type,offset.pointer[index_in].context), array_in, length_in);
}

template <typename T>
void Parameter::LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<T>> array_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in)
{
    for (unsigned i = 0u; i < lengthElements_in; i++)
    {
        LoadData(indexes_in.pointer[i], array_in.pointer[i], lengthArray_in.pointer[i]);
    }
}

template <typename T>
Pointer<T> Parameter::GetPointer(unsigned index)
{
    if (isValid == false)
    {
        std::cout << "Error: Pointer is not initialized.";
        return Pointer<T>();
    }
    return Pointer<T>((T *)offset.pointer[index].pointer, offset.pointer[index].type, offset.pointer[index].context);
}

#endif