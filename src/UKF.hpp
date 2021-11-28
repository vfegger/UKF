#ifndef UKF_HEADER
#define UKF_HEADER

#include "State.hpp"
#include "Measure.hpp"
#include "Input.hpp"
#include "Output.hpp"
#include "MathCustom.hpp"

class UKF
{
private:
    Input* input;
    Output* output;
public:
    UKF();
    ~UKF();
    void Initialize(Input* input);
    void Solve();
    void Export(Output* output);
    virtual void SigmaPointsGenerator(State* state, Point* &sigmaPoints, unsigned& sigmaLength);
};
#endif