// nvcc --shared --compiler-options -fPIC -o libjnet.so jnet.cu 

#include <stdio.h>
#include <cuda_runtime.h>
#define BLK 128
#define THR 128

static void checkError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void _reluforw(float *y, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] < 0) y[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" void reluforw(float *y, int n)
{
  _reluforw<<<BLK,THR>>>(y, n);
  checkError();
}

__global__ void _reluback(float *dy, float *y, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] <= 0) dy[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" void reluback(float *dy, float *y, int n)
{
  _reluback<<<BLK,THR>>>(dy, y, n);
  checkError();
}

__global__ void _softback(float *dy, float *y, int nrows, int ncols)
{
  float y0, sum;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    y0 = -INFINITY;
    //y0 = y[i0];
    for (int i=i0; i<i1; i++) {
      if (y[i] > y0) {
	y0 = y[i];
      }
    }
    sum = 0;
    for (int i=i0; i<i1; i++) {
      y[i] = exp(y[i]-y0);
      sum += y[i];
    }
    for (int i=i0; i<i1; i++) {
      y[i] /= sum;
      dy[i] = (y[i] - dy[i]) / ncols;
    }
    col += blockDim.x * gridDim.x;
  }
}

extern "C" void softback(float *dy, float *y, int nrows, int ncols)
{
  _softback<<<BLK,THR>>>(dy, y, nrows, ncols);
  checkError();
}

