# TODO
# make it work without cudart and gpu
# gpu: reluforw, reluback, softback
# dropout
# update
# careful with alloc-free of gpu variables
# do we need all the declarations, or should we keep the code generic


### Union types to cover both regular and cuda arrays:
using CUDArt: CudaArray, CudaMatrix, CudaVector
typealias Mat{t} Union(AbstractMatrix{t}, CudaMatrix{t})
typealias Vec{t} Union(AbstractVector{t}, CudaVector{t})

### Defaults for regular arrays:
import CUDArt: to_host, free, device_reset
to_host(x)=x
free(x)=x

### We need ones, zeros, similar for CudaArrays:
import Base: ones, zeros, similar
ones(x::Array,dims...)=ones(eltype(x),dims...)
ones{T}(a::CudaArray{T})=CudaArray(ones(T,size(a)))
ones{T}(a::CudaArray{T},dims...)=CudaArray(ones(T,dims...))
zeros(x::Array)=zeros(eltype(x),size(x))
zeros{T}(a::CudaArray{T})=CudaArray(zeros(T,size(a)))
zeros{T}(a::CudaArray{T},dims...)=CudaArray(zeros(T,dims...))
similar{T}(a::CudaArray{T}, dims::Int...) = similar(a, T, dims)


### We need gemm! and ger! for CudaArrays:
import Base.LinAlg.BLAS: gemm!, ger!, gemv!, axpy!
blas_set_num_threads(12)
using Base.LinAlg: BlasChar, BlasInt
libcublas="libcublas"

## gemm: C = alpha A*B + beta C
for (fname, elty) in
        (("cublasDgemm",:Float64),
         ("cublasSgemm",:Float32),
         ("cublasZgemm",:Complex128),
         ("cublasCgemm",:Complex64))
    @eval begin
        function gemm!(transA::BlasChar, transB::BlasChar, alpha::($elty), 
                       A::Mat{$elty}, B::Mat{$elty}, beta::($elty), C::Mat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch())
            end
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dB = isa(B, CudaArray) ? B : CudaArray(B)
            dC = isa(C, CudaArray) ? C : CudaArray(C)
            ccall(($(fname), $(libcublas)), Void,
                  (BlasChar, BlasChar, BlasInt, BlasInt, BlasInt, $elty, Ptr{$elty}, BlasInt, 
                  Ptr{$elty}, BlasInt, $elty, Ptr{$elty}, BlasInt),
                  transA, transB, m, n, k, alpha, pointer(dA), max(1,stride(A,2)), 
                  pointer(dB), max(1,stride(B,2)), beta, pointer(dC), max(1,stride(C,2)))
            if (!is(dA,A)) free(dA); end
            if (!is(dB,B)) free(dB); end
            if (!is(dC,C)) copy!(C,dC); free(dC); end
            C
        end
    end
end

### ger: A = α x y' + A
for (fname, elty) in (("cublasDger",:Float64),
                      ("cublasSger",:Float32),
                      ("cublasZger",:Complex128),
                      ("cublasCger",:Complex64))
    @eval begin
        function ger!(α::$elty, x::Vec{$elty}, y::Vec{$elty}, A::Mat{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch())
            n == length(y) || throw(DimensionMismatch())
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dx = isa(x, CudaArray) ? x : CudaArray(x)
            dy = isa(y, CudaArray) ? y : CudaArray(y)
            ccall(($(fname), $(libcublas)), Void,
                (BlasInt, BlasInt, $elty, Ptr{$elty},
                 BlasInt, Ptr{$elty}, BlasInt, Ptr{$elty}, BlasInt),
                 m, n, α, pointer(dx), 1, 
                 pointer(dy), 1, pointer(dA), max(1,stride(dA,2)))
            if (!is(dA,A)) copy!(A,dA); free(dA); end
            if (!is(dx,x)) free(dx); end
            if (!is(dy,y)) free(dy); end
            A
        end
    end
end


### gemv: y = alpha trans(A) x + beta y
for (fname, elty) in
        (("cublasDgemv",:Float64),
         ("cublasSgemv",:Float32),
         ("cublasZgemv",:Complex128),
         ("cublasCgemv",:Complex64))
    @eval begin
        function gemv!(trans::BlasChar, alpha::($elty), A::Mat{$elty}, X::Vec{$elty}, beta::($elty), Y::Vec{$elty})
            m,n = size(A)
            length(X) == (trans == 'N' ? n : m) && length(Y) == (trans == 'N' ? m : n) || throw(DimensionMismatch())
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dX = isa(X, CudaArray) ? X : CudaArray(X)
            dY = isa(Y, CudaArray) ? Y : CudaArray(Y)
            ccall(($(fname), $(libcublas)), Void,
                (BlasChar, BlasInt, BlasInt, $elty,
                 Ptr{$elty}, BlasInt, Ptr{$elty}, BlasInt,
                 $elty, Ptr{$elty}, BlasInt),
                 trans, size(dA,1), size(dA,2), alpha,
                 dA, max(1,stride(dA,2)), dX, stride(dX,1),
                 beta, dY, stride(dY,1))
            if (!is(dA,A)) free(dA); end
            if (!is(dX,X)) free(dX); end
            if (!is(dY,Y)) copy!(Y,dY); free(dY); end
            Y
        end
    end
end



### TODO: just implement softmax if we are not going to use the rest of NumExt
# using NumericExtensions: softmax!


### To help debug memory

function mysizeof(x)
    return sizeof(eltype(x))*length(x)
end

### CUDA versions of activation functions:

const libjnet="./libjnet.so.7"

function reluforw(y::CudaMatrix{Float32})
    ccall(("reluforw",libjnet), Void, (Ptr{Float32},Cint), y, length(y))
end

function reluback(dy::CudaMatrix{Float32}, y::CudaMatrix{Float32})
    ccall(("reluback",libjnet), Void, (Ptr{Float32},Ptr{Float32},Cint), dy, y, length(y))
end

function softback(dy::Mat{Float32}, y::CudaMatrix{Float32})
    ddy = isa(dy, CudaArray) ? dy : CudaArray(dy)
    ccall(("softback",libjnet), Void, (Ptr{Float32},Ptr{Float32}, Cint,Cint), ddy, y, size(y,1), size(y,2))
    if (!is(ddy,dy)) free(ddy); end
end

