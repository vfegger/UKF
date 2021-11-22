#include "Output.hpp"

Output::Output(){
    outputParametersLength = 0u;
    outputDataLength = 0u;
    parameters = NULL;
    data = NULL;
}

Output::~Output(){
    outputParametersLength = 0u;
    outputDataLength = 0u;
    if (parameters != NULL){
        delete[] parameters;
    }
    if (data != NULL){
        delete[] data;
    }
}

void Output::Export(std::string &out, std::string &ext, std::string &delimiter, bool concatOutput){
    if (concatOutput){
        for(unsigned i = 0u; i < outputParametersLength; i++){
            std::string fileName(out+parameters[i].GetName()+ext);
            std::ofstream file(fileName);
            if (file.is_open()){
                std::cout << "Exporting file " << fileName << " ...\n";
                unsigned len = parameters[i].GetLength();
                for(unsigned j = 0; j < len; j++){
                    file << parameters[i][j] << delimiter;
                }
                file.close();
                std::cout << "\tCompleted exporting file " << fileName << "\n";
            } else {
                std::cout << "Failed exporting file " << fileName << "\n";
            }
        }
    } else {
        std::string fileName(out+ext);
        std::ofstream file(fileName);
        if (file.is_open()){
            std::cout << "Exporting file " << fileName << " ...\n";
            for(unsigned i = 0u; i < outputParametersLength; i++){
                unsigned len = parameters[i].GetLength();
                for(unsigned j = 0; j < len; j++){
                    file << parameters[i][j] << delimiter;
                }
                file.close();
            }
            std::cout << "\tCompleted exporting file " << fileName << "\n";
        } else {
                std::cout << "Failed exporting file " << fileName << "\n";
        }
    }
}