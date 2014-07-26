#include "mex.h"

/* The computational routine */
void poly3basis(double *x, double *z, mwSize n)
{
  mwSize i, j, k, l;
  l = 0;
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      for (k=j; k<n; k++) {
	z[l++] = x[i]*x[j]*x[k];
      }
    }
  }
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  double *inMatrix;               /* Nx1 input matrix */
  size_t ncols;                   /* size of matrix */
  double *outMatrix;              /* output matrix */

  /* check for proper number of arguments */
  if(nrhs!=1) {
    mexErrMsgIdAndTxt("MyToolbox:poly3basis:nrhs","One input required.");
  }
  if(nlhs!=1) {
    mexErrMsgIdAndTxt("MyToolbox:poly3basis:nlhs","One output required.");
  }
    
  /* make sure the input argument is type double */
  if( !mxIsDouble(prhs[0]) || 
      mxIsComplex(prhs[0])) {
    mexErrMsgIdAndTxt("MyToolbox:poly3basis:notDouble","Input matrix must be type double.");
  }
    
  /* check that number of rows in second input argument is 1 */
  if(mxGetN(prhs[0])!=1) {
    mexErrMsgIdAndTxt("MyToolbox:poly3basis:notColVector","Input must be a column vector.");
  }
    
  /* create a pointer to the real data in the input matrix  */
  inMatrix = mxGetPr(prhs[0]);

  /* get dimensions of the input matrix, and compute size of output */
  mwSize n = mxGetM(prhs[0]);
  mwSize o = n*(n+1)*(n+2)/6;

  /* create the output matrix */
  plhs[0] = mxCreateDoubleMatrix((mwSize)o,1,mxREAL);

  /* get a pointer to the real data in the output matrix */
  outMatrix = mxGetPr(plhs[0]);

  /* call the computational routine */
  poly3basis(inMatrix,outMatrix,(mwSize)n);
}
