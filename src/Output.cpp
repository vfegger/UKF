#include "Output.hpp"

Output::Output(){
    outputParametersLength = 0u;
    outputDataStateLength = 0u;
    outputDataMeasureLength = 0u;
    parameters = NULL;
    data_state = NULL;
    data_state_covariance = NULL;
    data_measure = NULL;
    data_measure_covariance = NULL;
}

Output::~Output(){
    outputParametersLength = 0u;
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

void Output::SetOutput(Parameters* parameters_in, Data* data_state_in, Data* data_state_covariance_in, Data* data_measure_in, Data* data_measure_covariance_in, unsigned outputParametersLength_in, unsigned outputDataStateLength_in, unsigned outputDataMeasureLength_in){
    outputParametersLength = outputParametersLength_in;
    outputDataStateLength = outputDataStateLength_in;
    outputDataMeasureLength = outputDataMeasureLength_in;
    parameters = new(std::nothrow) Parameters[outputParametersLength];
    data_state = new(std::nothrow) Data[outputDataStateLength];
    data_state_covariance = new(std::nothrow) Data[outputDataStateLength];
    data_measure = new(std::nothrow) Data[outputDataMeasureLength];
    data_measure_covariance = new(std::nothrow) Data[outputDataMeasureLength];
    for(unsigned i = 0; i < outputParametersLength; i++){
        parameters[i] = Parameters(parameters_in[i]);
    }
    for(unsigned i = 0; i < outputDataStateLength; i++){
        data_state[i] = Data(data_state_in[i]);
        data_state_covariance[i] = Data(data_state_covariance_in[i]);
    }
    for(unsigned i = 0; i < outputDataMeasureLength; i++){
        data_measure[i] = Data(data_measure_in[i]);
        data_measure_covariance[i] = Data(data_measure_covariance_in[i]);
    }
}

void Output::ExportDivide(Parameters* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter){
    for(unsigned i = 0u; i < length; i++){
        unsigned sizeType = parameters[i].GetSizeType();
        std::string fileName(out+parameters[i].GetName()+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            unsigned len = parameters[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                if(sizeType == 1){
                file << parameters[i].GetValue<T>(j) << delimiter;
                } else if(sizeType == 4) {
                    
                }
            }
            file.close();
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
            std::cout << "Failed exporting file " << fileName << "\n";
            file.close();
        }
    }
}

void Output::ExportConcat(Parameters* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter){
    std::string fileName(out+ext);
    std::ofstream file(fileName);
    if (file.is_open()){
        std::cout << "Exporting file " << fileName << " ...\n";
        for(unsigned i = 0u; i < length; i++){
            unsigned len = parameters[i].GetLength();
            for(unsigned j = 0; j < len; j++){
                file << parameters[i].GetValue<T>(j) << delimiter;
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
        ExportConcat(parameters, outputParametersLength, out, ext, delimiter);
        ExportConcat(data_state, outputDataStateLength, out, ext, delimiter);
        ExportConcat(data_state_covariance, outputDataStateLength, out, ext, delimiter);
        ExportConcat(data_measure, outputDataMeasureLength, out, ext, delimiter);
        ExportConcat(data_measure_covariance, outputDataMeasureLength, out, ext, delimiter);
    } else {
        ExportDivide(parameters, outputParametersLength, out, ext, delimiter);
        ExportDivide(data_state, outputDataStateLength, out, ext, delimiter);
        ExportDivide(data_state_covariance, outputDataStateLength, out, ext, delimiter);
        ExportDivide(data_measure, outputDataMeasureLength, out, ext, delimiter);
        ExportDivide(data_measure_covariance, outputDataMeasureLength, out, ext, delimiter);        
    }    
}