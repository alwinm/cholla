#include <math.h>
#include <iostream>

#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"

#include "reduce.h"

#define SIMB 512 // approx simultaneous blocks running

// Test multiple reduction strategies

__global__ void atomic_reduce_kernel(Real *dev_conserved, int n_cells, Real *dev_array)
{
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  Real d = dev_conserved[id];
  // For a given block, a contiguous TPB section of dev_array will be filled by atomicAdd
  atomicAdd(&dev_array[blockIdx.x%SIMB*TPB + threadIdx.x], d);

}


__global__ void atomic_reduce_kernel2(Real *dev_conserved, int n_cells, Real *dev_array)
{
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  Real d = dev_conserved[id];
  // For a given block, a contiguous TPB section of dev_array will be filled by atomicAdd
  if (id < n_cells) atomicAdd(&dev_array[id%(SIMB*TPB)], d);

}

void atomic_reduce_host(Real *dev_conserved, int n_cells)
{
  int ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  Real *dev_array;
  Real host_array[SIMB*TPB];
  Real total = 0;
  Real time_start;
  Real time_kernel = 0;
  Real time_kernel2 = 0;
  Real time_memcpy = 0;
  Real time_loop = 0;
  Real time_total = 0;
  CudaSafeCall( cudaMalloc ( &dev_array, SIMB*TPB*sizeof(Real)));
  CudaSafeCall( cudaMemset ( &dev_array, 0, SIMB*TPB*sizeof(Real)));

  for (int j=0; j<1000; j++) {
    time_start = get_time();
    
    hipLaunchKernelGGL(atomic_reduce_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();
    time_kernel += get_time() - time_start;
    
    CudaSafeCall( cudaMemcpy(host_array, dev_array, SIMB*TPB*sizeof(Real), cudaMemcpyDeviceToHost) );
    time_memcpy += get_time() - time_start;
    
    for (int i=0; i<SIMB*TPB; i++) {
      total += host_array[i];
    }

    time_loop += get_time() - time_start;
    
    time_total += get_time() - time_start;
    
    hipLaunchKernelGGL(atomic_reduce_kernel2, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();

    time_kernel2 += get_time() - time_start;
  }
  
  cudaFree(dev_array);

  time_kernel2 -= time_total;
  time_loop -= time_memcpy;
  time_memcpy -= time_kernel;

  time_total  *= 1000;
  time_loop   *= 1000;
  time_memcpy *= 1000;
  time_kernel *= 1000;
  time_kernel2 *= 1000;


  printf("Atomic_reduce Kernel: %9.4f Kernel2: %9.4f Memcpy: %9.4f Loop: %9.4f Total: %9.4f \n",
	 time_kernel,time_kernel2,time_memcpy,time_loop,time_total);
  
}


__global__ void shared_reduce_kernel2(Real *dev_conserved, int n_cells, Real *dev_array)
{
  __shared__ Real total[TPB];
  total[threadIdx.x] = 0;
  __syncthreads();
  
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int tid = threadIdx.x;
  Real d=0;
  if (id < n_cells) d = dev_conserved[id];

  total[tid] += d;

  __syncthreads();

  for (unsigned int s=1; s<blockDim.x; s*=2) {                                                           
    if (tid % (2*s) == 0) {
      total[tid] += total[tid+s];
    }                                                                                                    
    __syncthreads();                                                                                     
  }   

  if (tid == 0) dev_array[blockIdx.x] = total[0];
}

__global__ void shared_reduce_kernel(Real *dev_conserved, int n_cells, Real *dev_array)
{
  __shared__ Real total[TPB];
  total[threadIdx.x] = 0;
  __syncthreads();
  
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int tid = threadIdx.x;
  Real d = 0;
  if (id < n_cells) d = dev_conserved[id];

  total[tid] += d;

  __syncthreads();

  if (TPB >= 256) { if (tid < 128) { total[tid] += total[tid + 128]; } __syncthreads(); }
  if (TPB >= 128) { if (tid <  64) { total[tid] += total[tid +  64]; } __syncthreads(); }
  if (tid > 32) return;
  if (TPB >=  64) total[tid] += total[tid+32];
  if (TPB >=  32) total[tid] += total[tid+16];
  if (TPB >=  16) total[tid] += total[tid+ 8];
  if (TPB >=   8) total[tid] += total[tid+ 4];
  if (TPB >=   4) total[tid] += total[tid+ 2];
  if (TPB >=   2) total[tid] += total[tid+ 1];
  
  if (tid == 0) dev_array[blockIdx.x] = total[0];
}

void shared_reduce_host(Real *dev_conserved, int n_cells)
{
  int ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  int dev_size = ngrid;
  Real *dev_array;
  Real host_array[dev_size];
  Real total = 0;
  Real time_start;
  Real time_kernel = 0;
  Real time_kernel2 = 0;
  Real time_memcpy = 0;
  Real time_loop = 0;
  Real time_total = 0;



  
  CudaSafeCall( cudaMalloc ( &dev_array, dev_size*sizeof(Real)));
  CudaSafeCall( cudaMemset ( &dev_array, 0, dev_size*sizeof(Real)));

  for (int j=0; j<1000; j++) {
    time_start = get_time();
    
    hipLaunchKernelGGL(shared_reduce_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();
    time_kernel += get_time() - time_start;
    
    CudaSafeCall( cudaMemcpy(host_array, dev_array, dev_size*sizeof(Real), cudaMemcpyDeviceToHost) );
    time_memcpy += get_time() - time_start;
    
    for (int i=0; i<dev_size; i++) {
      total += host_array[i];
    }

    time_loop += get_time() - time_start;
    
    time_total += get_time() - time_start;
    
    hipLaunchKernelGGL(shared_reduce_kernel2, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();

    time_kernel2 += get_time() - time_start;
  }
  
  cudaFree(dev_array);

  time_kernel2 -= time_total;
  time_loop -= time_memcpy;
  time_memcpy -= time_kernel;

  time_total  *= 1000;
  time_loop   *= 1000;
  time_memcpy *= 1000;
  time_kernel *= 1000;
  time_kernel2 *= 1000;


  printf("Shared_reduce Kernel: %9.4f Kernel2: %9.4f Memcpy: %9.4f Loop: %9.4f Total: %9.4f \n",
	 time_kernel,time_kernel2,time_memcpy,time_loop,time_total);
  
}




__global__ void one_reduce_kernel(Real *dev_conserved, int n_cells, Real *dev_array)
{
  __shared__ Real total[TPB];
  total[threadIdx.x] = 0;
  __syncthreads();
  
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  //int id = threadIdx.x + blockId * blockDim.x;
  int tid = threadIdx.x;
  //Real d = dev_conserved[id];
  Real subtotal = 0;
  for (int i=tid; i<n_cells; i+=TPB){
    subtotal += dev_conserved[i];
  }
  total[tid] = subtotal;
  

  
  //total[tid] += d;

  __syncthreads();

  if (TPB >= 256) { if (tid < 128) { total[tid] += total[tid + 128]; } __syncthreads(); }
  if (TPB >= 128) { if (tid <  64) { total[tid] += total[tid +  64]; } __syncthreads(); }
  if (tid > 32) return;
  if (TPB >=  64) total[tid] += total[tid+32];
  if (TPB >=  32) total[tid] += total[tid+16];
  if (TPB >=  16) total[tid] += total[tid+ 8];
  if (TPB >=   8) total[tid] += total[tid+ 4];
  if (TPB >=   4) total[tid] += total[tid+ 2];
  if (TPB >=   2) total[tid] += total[tid+ 1];
  
  if (tid == 0) dev_array[blockIdx.x] = total[0];
}

void one_reduce_host(Real *dev_conserved, int n_cells)
{
  int ngrid = 1;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  int dev_size = ngrid;
  Real *dev_array;
  Real host_array[dev_size];
  Real total = 0;
  Real time_start;
  Real time_kernel = 0;
  Real time_kernel2 = 0;
  Real time_memcpy = 0;
  Real time_loop = 0;
  Real time_total = 0;



  
  CudaSafeCall( cudaMalloc ( &dev_array, dev_size*sizeof(Real)));
  CudaSafeCall( cudaMemset ( &dev_array, 0, dev_size*sizeof(Real)));

  for (int j=0; j<100; j++) {
    time_start = get_time();
    
    hipLaunchKernelGGL(one_reduce_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();
    time_kernel += get_time() - time_start;
    
    CudaSafeCall( cudaMemcpy(host_array, dev_array, dev_size*sizeof(Real), cudaMemcpyDeviceToHost) );
    time_memcpy += get_time() - time_start;
    
    for (int i=0; i<dev_size; i++) {
      total += host_array[i];
    }

    time_loop += get_time() - time_start;
    
    time_total += get_time() - time_start;
    
    //hipLaunchKernelGGL(shared_reduce_kernel2, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();

    time_kernel2 += get_time() - time_start;
  }
  
  cudaFree(dev_array);

  time_kernel2 -= time_total;
  time_loop -= time_memcpy;
  time_memcpy -= time_kernel;

  time_total  *= 10000;
  time_loop   *= 10000;
  time_memcpy *= 10000;
  time_kernel *= 10000;
  time_kernel2 *= 10000;


  printf("One_reduce Kernel: %9.4f Kernel2: %9.4f Memcpy: %9.4f Loop: %9.4f Total: %9.4f \n",
	 time_kernel,time_kernel2,time_memcpy,time_loop,time_total);
  
}



__global__ void gloop_reduce_kernel(Real *dev_conserved, int n_cells, Real *dev_array)
{
  __shared__ Real total[TPB];
  total[threadIdx.x] = 0;
  __syncthreads();

  // There are SIMB blocks each with TPB threads
  // Each block contributes 1 number to dev_array[blockIdx.x]
  
  
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int tid = threadIdx.x;
  //Real d = dev_conserved[id];
  Real subtotal = 0;
  for (int i=id; i<n_cells; i+=TPB*SIMB){
    subtotal += dev_conserved[i];
  }
  total[tid] = subtotal;
  

  
  //total[tid] += d;

  __syncthreads();

  if (TPB >= 256) { if (tid < 128) { total[tid] += total[tid + 128]; } __syncthreads(); }
  if (TPB >= 128) { if (tid <  64) { total[tid] += total[tid +  64]; } __syncthreads(); }
  if (tid > 32) return;
  if (TPB >=  64) total[tid] += total[tid+32];
  if (TPB >=  32) total[tid] += total[tid+16];
  if (TPB >=  16) total[tid] += total[tid+ 8];
  if (TPB >=   8) total[tid] += total[tid+ 4];
  if (TPB >=   4) total[tid] += total[tid+ 2];
  if (TPB >=   2) total[tid] += total[tid+ 1];
  
  if (tid == 0) dev_array[blockIdx.x] = total[0];
}

void gloop_reduce_host(Real *dev_conserved, int n_cells)
{
  int ngrid = SIMB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  int dev_size = ngrid;
  Real *dev_array;
  Real host_array[dev_size];
  Real total = 0;
  Real time_start;
  Real time_kernel = 0;
  Real time_kernel2 = 0;
  Real time_memcpy = 0;
  Real time_loop = 0;
  Real time_total = 0;



  
  CudaSafeCall( cudaMalloc ( &dev_array, dev_size*sizeof(Real)));
  CudaSafeCall( cudaMemset ( &dev_array, 0, dev_size*sizeof(Real)));

  for (int j=0; j<1000; j++) {
    time_start = get_time();
    
    hipLaunchKernelGGL(gloop_reduce_kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();
    time_kernel += get_time() - time_start;
    
    CudaSafeCall( cudaMemcpy(host_array, dev_array, dev_size*sizeof(Real), cudaMemcpyDeviceToHost) );
    time_memcpy += get_time() - time_start;
    
    for (int i=0; i<dev_size; i++) {
      total += host_array[i];
    }

    time_loop += get_time() - time_start;
    
    time_total += get_time() - time_start;
    
    //hipLaunchKernelGGL(shared_reduce_kernel2, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, dev_array);
    cudaDeviceSynchronize();

    time_kernel2 += get_time() - time_start;
  }
  
  cudaFree(dev_array);

  time_kernel2 -= time_total;
  time_loop -= time_memcpy;
  time_memcpy -= time_kernel;

  time_total  *= 1000;
  time_loop   *= 1000;
  time_memcpy *= 1000;
  time_kernel *= 1000;
  time_kernel2 *= 1000;


  printf("Gloop_reduce Kernel: %9.4f Kernel2: %9.4f Memcpy: %9.4f Loop: %9.4f Total: %9.4f \n",
	 time_kernel,time_kernel2,time_memcpy,time_loop,time_total);
  
}
