#pragma once

//#include <math.h>
//#include <iostream>

//#include "../utils/gpu.hpp"
#include "../global/global.h"
//#include "../global/global_cuda.h"




void atomic_reduce_host(Real *dev_conserved, int n_cells);
void shared_reduce_host(Real *dev_conserved, int n_cells);
void    one_reduce_host(Real *dev_conserved, int n_cells);
void  gloop_reduce_host(Real *dev_conserved, int n_cells);
