#ifndef POINT_COVARIANCE_HEADER
#define POINT_COVARIANCE_HEADER

#include "Data.hpp"

class PointCovariance
{
private:
    Data* dataCovariance;
    double* stateCovariance;
    bool* compactForm;
    unsigned length_data;
    unsigned length_state;
public:
    PointCovariance();
    PointCovariance(Data* data_input, Data* dataCovariance_input, unsigned length_input);
    ~PointCovariance();
    void UpdateArrayFromData();
    void UpdateDataFromArray();
    unsigned GetLengthState();
    unsigned GetLengthData();
    Data* GetDataCovariance();
    double* GetStateCovariance();
};
#endif