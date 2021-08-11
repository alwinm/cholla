#ifdef CUDA
#include"gpu.hpp"
#include"global.h"
#include"global_cuda.h"




__global__ void PackBuffers3DKernel(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void PackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

__global__ void UnpackBuffers3DKernel(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void UnpackBuffers3D(Real * buffer, Real * c_head, int isize, int jsize, int ksize, int nx, int ny, int idxoffset, int offset, int n_fields, int n_cells);

void PackGhostCells(Real * c_head,
		    int nx, int ny, int n_fields, int n_cells, int flags[],
		    int isize, int jsize, int ksize,
		    int imin, int jmin, int kmin);

__global__ void PackGhostCellsKernel(Real * c_head,
				     int nx, int ny, int n_fields, int n_cells,
				     int f0, int f1, int f2, int f3, int f4, int f5,
				     int isize, int jsize, int ksize,
				     int imin, int jmin, int kmin);

__device__ int SetBoundaryMapping(int ig, int jg, int kg, Real *a, int flags[]);

__device__ int FindIndex(int ig, int nx, int flag, int face, int n_ghost, Real *a);


#endif
