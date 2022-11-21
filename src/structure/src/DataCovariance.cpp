#include "../include/DataCovariance.hpp"

DataCovariance::DataCovariance()
{
    lengthElements = 0u;
    names = Pointer<std::string>(PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = Pointer<unsigned>(PointerType::CPU, PointerContext::CPU_Only);
    offset = Pointer<Pointer<double>>(PointerType::CPU, PointerContext::CPU_Only);
    offsetArray = Pointer<unsigned>(PointerType::CPU, PointerContext::CPU_Only);
    pointer = Pointer<double>();
    length = 0u;
    count = 0u;
    isValid = false;
}
DataCovariance::DataCovariance(unsigned lengthElements_in)
{
    lengthElements = lengthElements_in;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offsetArray = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || offsetArray.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    pointer = Pointer<double>();
    length = 0u;
    isValid = false;
    count = 0u;
}
DataCovariance::DataCovariance(const Data &data_in)
{
    lengthElements = data_in.GetCapacity();
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offsetArray = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || offsetArray.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    count = data_in.GetCount();
    unsigned offset_aux = 0u;
    for (unsigned i = 0u; i < count; i++)
    {
        offset.pointer[i] = Pointer<double>();
        names.pointer[i] = data_in.GetNames(i);
        lengthOffset.pointer[i] = data_in.GetLength(i);
        offsetArray.pointer[i] = offset_aux;
        offset_aux += lengthOffset.pointer[i];
    }
    for (unsigned i = count; i < lengthElements; i++)
    {
        offset.pointer[i] = Pointer<double>();
        names.pointer[i] = "";
        lengthOffset.pointer[i] = 0u;
        offsetArray.pointer[i] = offset_aux;
        offset_aux += lengthOffset.pointer[i];
    }
    isValid = data_in.GetValidation();
    if (isValid)
    {
        length = data_in.GetLength();
        PointerType type = data_in.GetPointer().type;
        PointerContext context = data_in.GetPointer().context;
        pointer = MemoryHandler::Alloc<double>(length * length, type, context);
        if (pointer.pointer == NULL)
        {
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        for (unsigned i = 0u; i < count; i++)
        {
            offset.pointer[i] = Pointer<double>(pointer.pointer + offsetArray.pointer[i] * (length + 1u), type, context);
        }
        MemoryHandler::Set(pointer, 0.0, 0, length * length);
    }
}
DataCovariance::DataCovariance(const DataCovariance &dataCovariance_in)
{
    pointer = Pointer<double>();
    length = 0u;
    lengthElements = dataCovariance_in.lengthElements;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offsetArray = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || offset.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    count = dataCovariance_in.count;
    for (unsigned i = 0u; i < lengthElements; i++)
    {
        offset.pointer[i] = Pointer<double>(dataCovariance_in.pointer.type, dataCovariance_in.pointer.context);
        names.pointer[i] = dataCovariance_in.names.pointer[i];
        lengthOffset.pointer[i] = dataCovariance_in.lengthOffset.pointer[i];
        offsetArray.pointer[i] = dataCovariance_in.offsetArray.pointer[i];
    }
    isValid = dataCovariance_in.isValid;
    if (isValid)
    {
        length = dataCovariance_in.length;
        pointer = MemoryHandler::Alloc<double>(length * length, dataCovariance_in.pointer.type, dataCovariance_in.pointer.context);
        if (pointer.pointer == NULL)
        {
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        for (unsigned i = 0u; i < count; i++)
        {
            offset.pointer[i] = Pointer<double>(pointer.pointer + offsetArray.pointer[i] * (length + 1u), pointer.type, pointer.context);
        }
        MemoryHandler::Copy(pointer, dataCovariance_in.pointer, length * length);
    }
}
DataCovariance& DataCovariance::operator=(const DataCovariance &dataCovariance_in)
{
    pointer = Pointer<double>();
    length = 0u;
    lengthElements = dataCovariance_in.lengthElements;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offsetArray = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || offset.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    count = dataCovariance_in.count;
    for (unsigned i = 0u; i < lengthElements; i++)
    {
        offset.pointer[i] = Pointer<double>(dataCovariance_in.pointer.type, dataCovariance_in.pointer.context);
        names.pointer[i] = dataCovariance_in.names.pointer[i];
        lengthOffset.pointer[i] = dataCovariance_in.lengthOffset.pointer[i];
        offsetArray.pointer[i] = dataCovariance_in.offsetArray.pointer[i];
    }
    isValid = dataCovariance_in.isValid;
    if (isValid)
    {
        length = dataCovariance_in.length;
        pointer = MemoryHandler::Alloc<double>(length * length, dataCovariance_in.pointer.type, dataCovariance_in.pointer.context);
        if (pointer.pointer == NULL)
        {
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        for (unsigned i = 0u; i < count; i++)
        {
            offset.pointer[i] = Pointer<double>(pointer.pointer + offsetArray.pointer[i] * (length + 1u), pointer.type, pointer.context);
        }
        MemoryHandler::Copy(pointer, dataCovariance_in.pointer, length * length);
    }
}
unsigned DataCovariance::Add(std::string name_in, unsigned length_in)
{
    isValid = false;
    if (count >= lengthElements)
    {
        std::cout << "Error: Added element is over the limit.\n";
        return lengthElements;
    }
    names.pointer[count] = name_in;
    lengthOffset.pointer[count] = length_in;
    offsetArray.pointer[count] = offsetArray.pointer[count - 1] + length_in;
    count++;
    return count - 1;
}
void DataCovariance::Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in)
{
    unsigned index = 0u;
    for (unsigned i = 0; i < lengthElements_in; i++)
    {
        index = Add(names_in.pointer[i], lengthArray_in.pointer[i]);
        if (indexes.pointer != NULL)
        {
            indexes.pointer[i] = index;
        }
    }
}
void DataCovariance::Initialize(PointerType type_in, PointerContext context_in)
{
    if (pointer.pointer != NULL)
    {
        MemoryHandler::Free(pointer);
    }
    length = 0u;
    for (unsigned i = 0u; i < count; i++)
    {
        length += lengthOffset.pointer[i];
    }
    pointer = MemoryHandler::Alloc<double>(length * length, type_in, context_in);
    if (pointer.pointer == NULL)
    {
        std::cout << "Error: Initialization wasn't successful.\n";
        return;
    }
    MemoryHandler::Set(pointer, 0.0, 0u, length);
    for (unsigned i = 0u; i < count; i++)
    {
        offset.pointer[i] = Pointer<double>(pointer.pointer + offsetArray.pointer[count] * (length + 1u), type_in, context_in);
    }
    isValid = true;
}
void DataCovariance::LoadData(unsigned index_in, Pointer<double> array_in, unsigned length_in, DataCovarianceMode mode_in)
{
    if (isValid == false)
    {
        std::cout << "Error: Load while structure is not initialized.\n";
        return;
    }
    if (index_in >= count)
    {
        std::cout << "Error: Index is out of range.\n";
        return;
    }
    switch (mode_in)
    {
    case DataCovarianceMode::Natural:
        MemoryHandler::Copy(offset.pointer[index_in], array_in, length_in * length_in);
        break;
    case DataCovarianceMode::Compact:
        // TODO: URGENT !!!
        for (unsigned i = 0; i < length_in; i++)
        {
            offset.pointer[index_in].pointer[i * length + i] = array_in.pointer[i];
        }
        break;
    case DataCovarianceMode::Complete:
        std::cout << "Warning: This mode overwrites the whole covariance matrix.\n";
        if (length != length_in)
        {
            std::cout << "Error: The dimensions of the covariance matrix and the covariance matrix input do not match.\n";
        }
        MemoryHandler::Copy(pointer, array_in, length_in * length_in);
        break;
    default:
        std::cout << "Error: Covariance mode is not implemented.\n";
        break;
    }
}
void DataCovariance::LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<double>> array_in, Pointer<unsigned> lengthArray_in, Pointer<DataCovarianceMode> modeArray_in, unsigned lengthElements_in)
{
    for (unsigned i = 0u; i < lengthElements_in; i++)
    {
        LoadData(indexes_in.pointer[i], array_in.pointer[i], lengthArray_in.pointer[i], modeArray_in.pointer[i]);
    }
}
unsigned DataCovariance::GetCapacity() const
{
    return lengthElements;
}
Pointer<double> DataCovariance::GetPointer() const
{
    if (isValid == false)
    {
        std::cout << "Error: Pointer is not initialized.\n";
        return Pointer<double>();
    }
    return pointer;
}
bool DataCovariance::GetValidation() const
{
    return isValid;
}
unsigned DataCovariance::GetCount() const
{
    return count;
}
unsigned DataCovariance::GetLength() const
{
    if (isValid == false)
    {
        std::cout << "Error: Length of pointer is not initialized.\n";
        return 0u;
    }
    return length;
}
unsigned DataCovariance::GetLength(unsigned index_in) const
{
    if (index_in >= count)
    {
        std::cout << "Error: Invalid index access.\n";
        return 0u;
    }
    return lengthOffset.pointer[index_in];
}
unsigned DataCovariance::GetOffset(unsigned index_in) const
{
    if (index_in >= count)
    {
        std::cout << "Error: Invalid index access.\n";
        return 0u;
    }
    return offsetArray.pointer[index_in];
}
std::string DataCovariance::GetNames(unsigned index_in) const
{
    if (index_in >= count)
    {
        std::cout << "Error: Invalid index access.\n";
        return "";
    }
    return names.pointer[index_in];
}
Pointer<double> DataCovariance::operator[](unsigned index)
{
    return offset.pointer[index];
}
DataCovariance::~DataCovariance()
{
    MemoryHandler::Free(pointer);
    MemoryHandler::Free(offset);
    MemoryHandler::Free(lengthOffset);
    MemoryHandler::Free(offsetArray);
    MemoryHandler::Free(names);

    lengthElements = 0;
    length = 0u;
    count = 0u;
    isValid = false;
}
