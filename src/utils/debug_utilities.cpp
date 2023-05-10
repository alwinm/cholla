#pragma once
#include "../global/global.h"
#include "../global/global_cuda.h"


// https://forums.developer.nvidia.com/t/best-way-to-report-memory-consumption-in-cuda/21042/2
void print_memory_usage() {
  size_t free_byte ;  
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  
  if ( cudaSuccess != cuda_status ){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(1);
  }
  double free_db = (double)free_byte ;

  double total_db = (double)total_byte ;

  double used_db = total_db - free_db ;

  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
	 used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
