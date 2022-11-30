#ifndef HELPERCUDA_HEADER
#define HELPERCUDA_HEADER

#include <cuda_runtime.h>

namespace HelperCUDA
{
    void Initialize(void *pointer_out, const void *value_in, unsigned length_in, unsigned size_in);
};

#endif