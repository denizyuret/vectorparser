function matMAD_matlab_docs
 
%
% this test below will time matrix multiply-add (MAD) on CPU vs on GPU
%
 
echo off
 
%sizze=10240
sizze=4096
A =rand( [sizze sizze] );
B = inv( A );
format short
 
% CPU
disp('CPU ::')
tic;
Acpu = A ;
Bcpu = B ;
% Multiply-add on CPU
Ccpu = Bcpu * Acpu - eye( size( Acpu ) );
% pull a result
minCcpu = abs(min(min( Ccpu )));
time_cpu=toc;
maxCcpu = abs(max(max( Ccpu )));
S=whos('Acpu');
sizeGB_cpu=getfield(S,'bytes')/1024/1024/1024;
 
% GPU
disp('GPU ::')
tic;
Agpu = gpuArray( A );
Bgpu = gpuArray( B );
Geye = gpuArray( eye(size( A )) );
% Multiply-add on GPU
Cgpu = ( Bgpu * Agpu - Geye );
% this forces the computation, and converts the result to standard Matlab variable
Cgpu = gather( Cgpu );
% pull a result
minCgpu = abs(min(min( Cgpu )));
time_gpu=toc;
maxCgpu = abs(max(max( Cgpu )));
 
echo on
disp('Time (s), CPU vs GPU')
[ time_cpu time_gpu ]
disp(' Matrix size (GB), on CPU ')
[ sizeGB_cpu ]
disp('Error (arbitrary units)')
[ abs(minCcpu-minCgpu)  abs(maxCcpu-maxCgpu) ]
 
return
 
