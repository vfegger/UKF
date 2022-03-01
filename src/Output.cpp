#include "Output.hpp"

Output::Output(){
    outputDataStateLength = 0u;
    outputDataMeasureLength = 0u;
    parameters = NULL;
    data_state = NULL;
    data_state_covariance = NULL;
    data_measure = NULL;
    data_measure_covariance = NULL;
}

Output::~Output(){
    outputDataStateLength = 0u;
    outputDataMeasureLength = 0u;
    if (parameters != NULL){
        delete[] parameters;
    }
    if (data_state != NULL){
        delete[] data_state;
    }
    if (data_state_covariance != NULL){
        delete[] data_state_covariance;
    }
    if (data_measure != NULL){
        delete[] data_measure;
    }
    if (data_measure_covariance != NULL){
        delete[] data_measure_covariance;
    }
}

void Output::SetOutput(Parameters* parameters_in, Data* data_state_in, Data* data_state_covariance_in, Data* data_measure_in, Data* data_measure_covariance_in, unsigned outputDataStateLength_in, unsigned outputDataMeasureLength_in){
    outputDataStateLength = outputDataStateLength_in;
    outputDataMeasureLength = outputDataMeasureLength_in;
    if(parameters_in != NULL){
        parameters = new(std::nothrow) Parameters(*parameters_in);
    } else {
        parameters = NULL;
    }
    data_state = new(std::nothrow) Data[outputDataStateLength];
    data_state_covariance = new(std::nothrow) Data[outputDataStateLength];
    data_measure = new(std::nothrow) Data[outputDataMeasureLength];
    data_measure_covariance = new(std::nothrow) Data[outputDataMeasureLength];
    for(unsigned i = 0; i < outputDataStateLength; i++){
        data_state[i] = Data(data_state_in[i]);
        data_state_covariance[i] = Data(data_state_covariance_in[i]);
    }
    for(unsigned i = 0; i < outputDataMeasureLength; i++){
        data_measure[i] = Data(data_measure_in[i]);
        data_measure_covariance[i] = Data(data_measure_covariance_in[i]);
    }
}

void Output::ExportDivide(Parameters* parameters, std::string &out, std::string &ext, std::string &delimiter){
    Parameters_Int* Int = parameters->Int;
    Parameters_UInt* UInt = parameters->UInt;
    Parameters_FP* FP = parameters->FP;
    for(unsigned i = 0u; i < parameters->GetLengthInt(); i++){
        std::string fileName(out+Int[i].GetName()+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            unsigned len = Int[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << Int[i][j] << delimiter;
            }
            file.close();
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
            std::cout << "Failed exporting file " << fileName << "\n";
            file.close();
        }
    }
    for(unsigned i = 0u; i < parameters->GetLengthUInt(); i++){
        std::string fileName(out+UInt[i].GetName()+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            unsigned len = UInt[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << UInt[i][j] << delimiter;
            }
            file.close();
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
            std::cout << "Failed exporting file " << fileName << "\n";
            file.close();
        }
    }
    for(unsigned i = 0u; i < parameters->GetLengthInt(); i++){
        std::string fileName(out+FP[i].GetName()+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            unsigned len = FP[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << FP[i][j] << delimiter;
            }
            file.close();
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
            std::cout << "Failed exporting file " << fileName << "\n";
            file.close();
        }
    }
}

void Output::ExportConcat(Parameters* parameters, std::string &out, std::string &ext, std::string &delimiter){
    Parameters_Int* Int = parameters->Int;
    Parameters_UInt* UInt = parameters->UInt;
    Parameters_FP* FP = parameters->FP;
    std::string fileNameInt(out+"Int"+ext);
    std::string fileNameUInt(out+"UInt"+ext);
    std::string fileNameFP(out+"FP"+ext);
    std::ofstream fileInt(fileNameInt);
    std::ofstream fileUInt(fileNameUInt);
    std::ofstream fileFP(fileNameFP);
    if (fileInt.is_open()){
        std::cout << "Exporting file " << fileNameInt << " ...\n";
        for(unsigned i = 0u; i < parameters->GetLengthInt(); i++){
            unsigned len = Int[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                fileInt << Int[i][j] << delimiter;
            }
            fileInt << "\n";
        }
        fileInt.close();
        std::cout << "\tCompleted exporting file " << fileNameInt << "\n";
    } else {
        std::cout << "Failed exporting file " << fileNameInt << "\n";
        fileInt.close();
    }
    if (fileUInt.is_open()){
        std::cout << "Exporting file " << fileNameUInt << " ...\n";
        for(unsigned i = 0u; i < parameters->GetLengthUInt(); i++){
            unsigned len = UInt[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                fileUInt << UInt[i][j] << delimiter;
            }
            fileUInt << "\n";
        }
        fileUInt.close();
        std::cout << "\tCompleted exporting file " << fileNameUInt << "\n";
    } else {
        std::cout << "Failed exporting file " << fileNameUInt << "\n";
        fileUInt.close();
    }
    if (fileFP.is_open()){
        std::cout << "Exporting file " << fileNameFP << " ...\n";
        for(unsigned i = 0u; i < parameters->GetLengthFP(); i++){
            unsigned len = FP[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                fileFP << FP[i][j] << delimiter;
            }
            fileFP << "\n";
        }
        fileFP.close();
        std::cout << "\tCompleted exporting file " << fileNameFP << "\n";
    } else {
        std::cout << "Failed exporting file " << fileNameFP << "\n";
        fileFP.close();
    }
}

void Output::ExportDivide(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter){
    for(unsigned i = 0u; i < length; i++){
        std::string fileName(out+data[i].GetName()+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            unsigned len = data[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << data[i][j] << delimiter;
            }
            file.close();
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
            std::cout << "Failed exporting file " << fileName << "\n";
            file.close();
        }
    }
}

void Output::ExportConcat(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter){
    std::string fileName(out+ext);
    std::ofstream file(fileName);
    if (file.is_open()){
        std::cout << "Exporting file " << fileName << " ...\n";
        for(unsigned i = 0u; i < length; i++){
            unsigned len = data[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << data[i][j] << delimiter;
            }
            file << "\n";
        }
        file.close();
        std::cout << "\tCompleted exporting file " << fileName << "\n";
    } else {
        std::cout << "Failed exporting file " << fileName << "\n";
        file.close();
    }
}

void Output::Export(std::string &out, std::string &ext, std::string &delimiter, bool concatOutput){
    if (concatOutput){
        ExportConcat(parameters, out, ext, delimiter);
        ExportConcat(data_state, outputDataStateLength, out, ext, delimiter);
        ExportConcat(data_state_covariance, outputDataStateLength, out, ext, delimiter);
        ExportConcat(data_measure, outputDataMeasureLength, out, ext, delimiter);
        ExportConcat(data_measure_covariance, outputDataMeasureLength, out, ext, delimiter);
    } else {
        ExportDivide(parameters, out, ext, delimiter);
        ExportDivide(data_state, outputDataStateLength, out, ext, delimiter);
        ExportDivide(data_state_covariance, outputDataStateLength, out, ext, delimiter);
        ExportDivide(data_measure, outputDataMeasureLength, out, ext, delimiter);
        ExportDivide(data_measure_covariance, outputDataMeasureLength, out, ext, delimiter);        
    }    
}