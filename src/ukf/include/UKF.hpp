#ifndef UKF_HEADER
#define UKF_HEADER

#include "../../math/include/Math.hpp"
#include "UKFMemory.hpp"

class UKF
{
private:

protected:
    UKFMemory* memory;
    
public:
    UKF(UKFMemory* memory_in);
    void Iterate();
};


#endif