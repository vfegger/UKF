#ifndef UKF_HEADER
#define UKF_HEADER

#include "UKFMemory.hpp"

class UKF
{
private:

protected:
    UKFMemory* memory;
    
public:
    void Iterate();
};


#endif