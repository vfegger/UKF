#include <iostream>
#include "structure/include/Data.hpp"
#include "structure/include/DataCovariance.hpp"
#include "structure/include/Parameter.hpp"
#include "parser/include/Parser.hpp"
#include "ukf/include/UKF.hpp"

class UKFMemory_Test : public UKFMemory{
public:
    UKFMemory_Test(Data& a, DataCovariance&b, DataCovariance&c, Data&d, DataCovariance&e, Parameter&f) : UKFMemory(a,b,c,d,e,f){

    }

    void Evolution(Data& data_inout, Parameter& parameter_in) override {

    }
    void Observation(Data& data_in, Parameter& parameter_in, Data& data_out) override {

    }
};

int main(){
    std::cout << "\nStart Execution\n\n";
    std::string path = "/mnt/d/Research/UKF/data/";

    Data a(10u);
    unsigned i = a.Add("Test_Data",3);
    a.Initialize();
    double b[3] = {1.1,2.2,3.3};
    a.LoadData(i, b, 3);

    std::cout << "Pointer: " << a.GetPointer() << "\n";
    for(unsigned j = 0u; j < 3; j++){
        std::cout << "Values: " << a[i][j] << "\n";
    }

    Parameter c(10u);
    unsigned k = c.Add("Test_Parm",3,sizeof(unsigned));
    c.Initialize();
    unsigned d[3] = {4u,5u,6u};
    c.LoadData(k,d,3);
    std::cout << "Pointer: " << c.GetPointer<unsigned>(k) << "\n";
    for(unsigned j = 0u; j < 3; j++){
        std::cout << "Values: " << c.GetPointer<unsigned>(k)[j] << "\n";
    }

    Parser parser(3u);
    unsigned index = parser.OpenFile(path + "TestData",".dat");
    std::string name = "";
    unsigned length = 0u;
    parser.ImportConfiguration(index,name,length);
    std::cout << "Name: " << name << "\n";
    std::cout << "Length: " << length << "\n";
    double values[length];
    parser.ImportData(index, length, values);
    for(unsigned j = 0u; j < length; j++){
        std::cout << "Values: " << values[j] << "\n";
    }
    DataCovariance e = DataCovariance(a);
    UKFMemory* ukfMemory = new UKFMemory_Test(a,e,e,a,e,c);

    UKF ukf(ukfMemory);

    delete ukfMemory;

    std::cout << "\nEnd Execution\n";
    return 0;
}