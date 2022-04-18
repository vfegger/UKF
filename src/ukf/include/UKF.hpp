#ifndef UKF_HEADER
#define UKF_HEADER

#include "../../math/include/Math.hpp"
#include "../../timer/include/Timer.hpp"
#include "UKFMemory.hpp"

#define UKF_TIMER 11u

class UKF
{
private:

protected:
    UKFMemory* memory;

    double alpha, beta, kappa;
    double lambda;
public:
    UKF(UKFMemory* memory_in, double alpha_in, double beta_in, double kappa_in);
    void Iterate(Timer& timer);
};


#endif