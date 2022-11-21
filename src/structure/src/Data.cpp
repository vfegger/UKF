#include "../include/Data.hpp"

Data::Data()
{
    lengthElements = 0;
    names = Pointer<std::string>(PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = Pointer<unsigned>(PointerType::CPU, PointerContext::CPU_Only);
    offset = Pointer<Pointer<double>>(PointerType::CPU, PointerContext::CPU_Only);
    pointer = Pointer<double>();
    length = 0u;
    count = 0u;
    isValid = false;

    multipleIndex = 0u;
    multipleLenght = 0u;
}
Data::Data(unsigned lengthElements_in)
{
    lengthElements = lengthElements_in;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    for (unsigned i = 0; i < lengthElements; i++)
    {
        names.pointer[i] = "";
        lengthOffset.pointer[i] = 0u;
        offset.pointer[i] = Pointer<double>();
    }
    pointer = Pointer<double>();
    length = 0u;
    isValid = false;
    count = 0u;

    multipleIndex = 0u;
    multipleLenght = 0u;
}
Data::Data(const Data &data_in)
{
    lengthElements = data_in.lengthElements;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    for (unsigned i = 0; i < lengthElements; i++)
    {
        names.pointer[i] = data_in.names.pointer[i];
        lengthOffset.pointer[i] = data_in.lengthOffset.pointer[i];
        offset.pointer[i] = Pointer<double>(data_in.offset.type, data_in.offset.context);
    }
    pointer = Pointer<double>();
    length = 0u;
    count = data_in.count;
    isValid = data_in.isValid;
    multipleIndex = 0u;
    multipleLenght = 0u;
    if (isValid)
    {
        length = data_in.length;
        pointer = MemoryHandler::Alloc<double>(length, data_in.pointer.type, data_in.pointer.context);
        if (pointer.pointer == NULL)
        {
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        MemoryHandler::Copy(pointer, data_in.pointer, length);
        unsigned offset_aux = 0u;
        for (unsigned i = 0; i < lengthElements; i++)
        {
            offset.pointer[i] = Pointer<double>(pointer.pointer + offset_aux, data_in.pointer.type, data_in.pointer.context);
            offset_aux += lengthOffset.pointer[i];
        }
    }
}
Data& Data::operator=(const Data &data_in)
{
    lengthElements = data_in.lengthElements;
    names = MemoryHandler::Alloc<std::string>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    lengthOffset = MemoryHandler::Alloc<unsigned>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<double>>(lengthElements, PointerType::CPU, PointerContext::CPU_Only);
    if (offset.pointer == NULL || lengthOffset.pointer == NULL || names.pointer == NULL)
    {
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    for (unsigned i = 0; i < lengthElements; i++)
    {
        names.pointer[i] = data_in.names.pointer[i];
        lengthOffset.pointer[i] = data_in.lengthOffset.pointer[i];
        offset.pointer[i] = Pointer<double>(data_in.offset.type, data_in.offset.context);
    }
    pointer = Pointer<double>();
    length = 0u;
    count = data_in.count;
    isValid = data_in.isValid;
    multipleIndex = 0u;
    multipleLenght = 0u;
    if (isValid)
    {
        length = data_in.length;
        pointer = MemoryHandler::Alloc<double>(length, data_in.pointer.type, data_in.pointer.context);
        if (pointer.pointer == NULL)
        {
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        MemoryHandler::Copy(pointer, data_in.pointer, length);
        unsigned offset_aux = 0u;
        for (unsigned i = 0; i < lengthElements; i++)
        {
            offset.pointer[i] = Pointer<double>(pointer.pointer + offset_aux, data_in.pointer.type, data_in.pointer.context);
            offset_aux += lengthOffset.pointer[i];
        }
    }
}
unsigned Data::Add(std::string name_in, unsigned length_in)
{
    isValid = false;
    if (count >= lengthElements)
    {
        std::cout << "Error: Added element is over the limit.\n";
        return lengthElements;
    }
    names.pointer[count] = name_in;
    lengthOffset.pointer[count] = length_in;
    count++;
    return count - 1;
}
void Data::Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in)
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
void Data::Initialize(PointerType type_in, PointerContext context_in)
{
    if (pointer.pointer != NULL && multipleIndex == 0u)
    {
        MemoryHandler::Free(pointer);
    }
    length = 0u;
    for (unsigned i = 0u; i < count; i++)
    {
        length += lengthOffset.pointer[i];
    }
    pointer = MemoryHandler::Alloc<double>(length, type_in, context_in);
    if (pointer.pointer == NULL)
    {
        std::cout << "Error: Initialization wasn't successful.\n";
        return;
    }
    MemoryHandler::Set(pointer, 0.0, 0u, length);
    unsigned offset_aux = 0u;
    for (unsigned i = 0u; i < count; i++)
    {
        offset.pointer[i] = Pointer<double>(pointer.pointer + offset_aux, type_in, context_in);
        offset_aux += lengthOffset.pointer[i];
    }
    isValid = true;
}
void Data::InstantiateMultiple(Pointer<Data> &dataArray_out, const Data &data_in, unsigned length_in, bool resetValues)
{
    if (length_in == 0u)
    {
        std::cout << "Error: Invalid size. It is expected at least 1.\n";
        return;
    }
    if (!data_in.GetValidation() && data_in.GetLength() == 0u)
    {
        std::cout << "Error: Invalid data for copy.\n";
    }
    if (dataArray_out.pointer != NULL)
    {
        MemoryHandler::Free(dataArray_out);
    }
    dataArray_out = MemoryHandler::Alloc<Data>(length_in, PointerType::CPU, PointerContext::CPU_Only);
    for (unsigned i = 0u; i < length_in; i++)
    {
        dataArray_out.pointer[i].lengthElements = data_in.lengthElements;
        dataArray_out.pointer[i].count = data_in.count;
        dataArray_out.pointer[i].lengthOffset = MemoryHandler::Alloc<unsigned>(data_in.lengthElements, PointerType::CPU, PointerContext::CPU_Only);
        dataArray_out.pointer[i].names = MemoryHandler::Alloc<std::string>(data_in.lengthElements, PointerType::CPU, PointerContext::CPU_Only);
        dataArray_out.pointer[i].offset = MemoryHandler::Alloc<Pointer<double>>(data_in.lengthElements, PointerType::CPU, PointerContext::CPU_Only);
        for (unsigned j = 0u; j < data_in.lengthElements; j++)
        {
            dataArray_out.pointer[i].names.pointer[j] = data_in.names.pointer[j];
            dataArray_out.pointer[i].lengthOffset.pointer[j] = data_in.lengthOffset.pointer[j];
        }
    }

    unsigned length_aux = length_in * data_in.GetLength();
    Pointer<double> pointer = MemoryHandler::Alloc<double>(length_aux,data_in.pointer.type,data_in.pointer.context);
    unsigned offset_aux = 0u;
    for (unsigned i = 0u; i < length_in; i++)
    {
        dataArray_out.pointer[i].length = data_in.GetLength();
        dataArray_out.pointer[i].pointer = Pointer<double>(pointer.pointer + offset_aux, data_in.pointer.type, data_in.pointer.context);
        for (unsigned j = 0u; j < data_in.lengthElements; j++)
        {
            dataArray_out.pointer[i].offset.pointer[j] = Pointer<double>(pointer.pointer + offset_aux, data_in.pointer.type, data_in.pointer.context);
            offset_aux += dataArray_out.pointer[i].lengthOffset.pointer[j];
        }
    }

    if (resetValues)
    {
        for (unsigned i = 0u; i < length_in; i++)
        {
            MemoryHandler::Set<double>(dataArray_out.pointer[i].pointer, 0.0, 0u, dataArray_out.pointer[i].length);
            dataArray_out.pointer[i].isValid = true;
            dataArray_out.pointer[i].multipleIndex = i;
            dataArray_out.pointer[i].multipleLenght = length_in;
        }
    }
    else
    {
        for (unsigned i = 0u; i < length_in; i++)
        {
            MemoryHandler::Copy(dataArray_out.pointer[i].pointer, data_in.pointer, dataArray_out.pointer[i].length);
            dataArray_out.pointer[i].isValid = true;
            dataArray_out.pointer[i].multipleIndex = i;
            dataArray_out.pointer[i].multipleLenght = length_in;
        }
    }
}
void Data::LoadData(unsigned index_in, Pointer<double> array_in, unsigned length_in)
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
    MemoryHandler::Copy(offset.pointer[index_in], array_in, length_in);
}
void Data::LoadData(Pointer<unsigned> indexes_in, Pointer<Pointer<double>> array_in, Pointer<unsigned> lengthArray_in, unsigned lengthElements_in)
{
    for (unsigned i = 0u; i < lengthElements_in; i++)
    {
        LoadData(indexes_in.pointer[i], array_in.pointer[i], lengthArray_in.pointer[i]);
    }
}
unsigned Data::GetCapacity() const
{
    return lengthElements;
}
Pointer<double> Data::GetPointer() const
{
    if (isValid == false)
    {
        std::cout << "Error: Pointer is not initialized.\n";
        return Pointer<double>();
    }
    return pointer;
}
bool Data::GetValidation() const
{
    return isValid;
}
unsigned Data::GetCount() const
{
    return count;
}
unsigned Data::GetLength() const
{
    if (isValid == false)
    {
        std::cout << "Error: Length of pointer is not initialized.\n";
        return 0u;
    }
    return length;
}
unsigned Data::GetLength(unsigned index_in) const
{
    if (index_in >= count)
    {
        std::cout << "Error: Invalid index access.\n";
        return 0u;
    }
    return lengthOffset.pointer[index_in];
}
std::string Data::GetNames(unsigned index_in) const
{
    if (index_in >= count)
    {
        std::cout << "Error: Invalid index access.\n";
        return "";
    }
    return names.pointer[index_in];
}
Pointer<double> &Data::operator[](unsigned index)
{
    return offset.pointer[index];
}
Data::~Data()
{
    if (multipleIndex == 0u)
    {
        MemoryHandler::Free(pointer);
    }
    MemoryHandler::Free(offset);
    MemoryHandler::Free(lengthOffset);
    MemoryHandler::Free(names);

    lengthElements = 0;
    length = 0u;
    count = 0u;
    isValid = false;

    multipleIndex = 0u;
    multipleLenght = 0u;
}