module Jnet

include("jnet_util.jl")

type Layer{t}
    fforw::Function	# y=forw(x)=fforw(w*x+b)
    fback::Function     # dx=back(dy)=w'*fback(dy)
    w::Mat{t}           # weight matrix
    b::Vec{t}           # bias vector

    dw::Mat{t}          # gradient wrt weight matrix
    dw1::Mat{t}         # moving average of gradients for momentum
    dw2::Mat{t}         # sum of squared gradients for adagrad

    db::Vec{t}          # gradient wrt bias vector
    db1::Vec{t}         # moving average of gradients for momentum
    db2::Vec{t}         # sum of squared gradients for adagrad

    x::Mat{t}           # last input
    y::Mat{t}           # last output
    dx::Mat{t}          # gradient wrt input
    dy::Mat{t}          # gradient wrt output
    xmask::Mat{t}       # input mask for dropout
    xones::Vec{t}       # vector of ones for bias calculation

    learningRate::t     # learning rate
    momentum::t         # momentum
    adagrad::Bool       # boolean indicating adagrad trick
    nesterov::Bool      # boolean indicating nesterov trick
    dropout::t          # probability of dropping inputs
    maxnorm::t          # parameter for maxnorm regularization
    L1::t               # parameter for L1 regularization
    L2::t               # parameter for L2 regularization

    Layer()=new()
    Layer(f,g,w,b)=new(f,g,w,b)
end


### Basic layer functions

function forw{t}(l::Layer{t}, x::Mat{t})
    initforw(l, x)					# alloc x,y,xones
    gemm!('N', 'N', one(t), l.w, l.x, zero(t), l.y)	# y=w*x
    if isdefined(l, :b)
	ger!(one(t), l.b, l.xones, l.y)			# y=y+b
    end
    if isdefined(l, :fforw)
	l.fforw(l.y)					# y=fforw(y)
    end
    return l.y
end

function back{t}(l::Layer{t}, dy::Mat{t}, dx::Bool=true)
    initback(l, dy, dx)		
    if isdefined(l, :fback)
	l.fback(l.dy, l.y)	     # this will overwrite l.dy and l.y!
    end 
    gemm!('N', 'T', one(t), l.dy, l.x, zero(t), l.dw)	# dw=dy*x'
    if isdefined(l, :b)
	gemv!('N', one(t), l.dy, l.xones, zero(t), l.db) # db=sum(dy,2)
    end
    if (dx) # dx is optional because it is expensive and unnecessary for input layer
        gemm!('T', 'N', one(t), l.w, l.dy, zero(t), l.dx) # dx=w'*dy
        return l.dx
    end
end


### Layer types

relu{t}(w::Mat{t},b::Vec{t}) = Layer{t}(reluforw, reluback, w, b)
soft{t}(w::Mat{t},b::Vec{t}) = Layer{t}(softforw, softback, w, b)

function reluforw{t}(y::Mat{t})
    for i=1:length(y)
        if (y[i] < zero(t))
            y[i] = zero(t)
        end
    end
    return y
end

function reluback{t}(dy::Mat{t}, y::Mat{t})
    for i=1:length(dy)
        if (y[i] <= zero(t))
            dy[i] = zero(t)
        end
    end
end

softforw(x)=x

function softback{t}(dy::Mat{t}, y::Mat{t})
    # we do softmax here instead of in forw
    # overwriting y from unnormalized log probabilities to normalized probabilities
    # NumericExtensions.softmax!(y,y,1) allocates unnecessary memory
    # dy is a 0-1 matrix of correct answers
    # will overwrite it with the gradient
    # TODO: is this a good interface?
    # TODO: other types of final layers, losses?

    for j=1:size(y,2)
        ymax = y[1,j]
        for i=2:size(y,1)
            if (y[i,j] > ymax)
                ymax = y[i,j]
            end
        end
        ysum = zero(t)
        for i=1:size(y,1)
            y[i,j] = exp(y[i,j] - ymax)
            ysum += y[i,j]
        end
        for i=1:size(y,1)
            y[i,j] /= ysum
            dy[i,j] = (y[i,j] - dy[i,j]) / size(y,2)
        end
    end
end


### A Net is just an array of Layers

typealias Net{t} Array{Layer{t},1}

function forw{t}(net::Net{t}, x::Mat{t})
    x = initforw(net, x)
    for i=1:length(net)
        x = forw(net[i], x)
    end
    return x
end

function back{t}(net::Net{t}, dy::Mat{t})
    dy = initback(net, dy)
    for i=length(net):-1:2
        dy = back(net[i], dy)
    end
    # No need to compute the last dx
    back(net[1], dy, false)
end


### Batch processing:

function forw{t}(net::Net{t}, x::Mat{t}, batch::Int)
    xrows,xcols = size(x)
    y = zeros(t, size(net[end].w, 1), xcols)
    info("forw:Alloc(y)=$(mysizeof(y))")
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        y[:,b:e] = to_host(forw(net, sub(x,1:xrows,b:e)))
    end
    return y
end

function forwback{t}(net::Net{t}, x::Mat{t}, labels::Vec{t}, batch::Int)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1),xcols
    y = zeros(t, yrows, ycols)
    info("forwback:Alloc(y)=$(mysizeof(y))")
    for i=1:length(labels)
        y[labels[i],i] = 1
    end
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        forw(net, sub(x,1:xrows,b:e))
        back(net, sub(y,1:yrows,b:e))
    end
end


### Memory management for forw and back
# This code is ugly because:
# 1. Need to free CudaArrays rather than relying on gc()
# 2. With getindex, ones, etc. missing from CUDArt, not possible to write generic code

function initmat{t}(l::Layer{t}, n::Symbol, dims::Dims, init::t=zero(t))
    if (!isdefined(l, n) || size(l.(n)) != dims)
	if (isdefined(l, n)) free(l.(n)); end
	l.(n) = similar(l.w, dims)
	fill!(l.(n), init)
        info("initmat($(n))=$(mysizeof(l.(n)))")
    end
end

function initforw{t}(net::Net{t}, x::Mat{t})
    l = net[1]
    if (isa(l.w, CudaArray))
	initmat(l, :x, size(x))
	copy!(l.x, x)
	x = l.x
    end
    return x
end

function initback{t}(net::Net{t}, dy::Mat{t})
    l = net[length(net)]
    if (isa(l.w, CudaArray))
	initmat(l, :dy, size(dy))
	copy!(l.dy, dy)
	dy = l.dy
    end
    return dy
end

function initforw{t}(l::Layer{t}, x::Mat{t})
    l.x = x
    if (!isdefined(l, :w))
        error("l.w not defined")
    end
    rows = size(l.w,1)
    cols = size(l.x,2)
    initmat(l, :y, (rows,cols))
    if (isdefined(l, :b))
	initmat(l, :xones, (cols,), one(t))
    end
end

function initback{t}(l::Layer{t}, dy::Mat{t}, dx::Bool)
    l.dy = dy
    initmat(l, :dw, size(l.w))
    if (isdefined(l, :b)) 
	initmat(l, :db, size(l.b))
    end
    if (dx) 
	initmat(l, :dx, size(l.x))
    end
end


### TEST CODE

using MAT

function load()
    file = matopen("dev.mat")
    data = read(file, "dev")
    close(file)
    return data
end

function test(data)
    # result = devices((x->true), nmax=1) do devlist
    # device(devlist[1])

    # info("Loading...")
    # data = load()

    info("Initializing...")
    x10k = data["x"][:,1:10000]
    y10k = data["y"][1:10000]
    w1 = data["w1"][:,2:end]
    b1 = data["w1"][:,1]
    w2 = data["w2"][:,2:end]
    b2 = data["w2"][:,1]

    n0 = [relu(w1,b1), soft(w2,b2)]
    info("Running 10k cpu test with bias...")
    info("CPU Forward 1")
    @time y1 = forw(n0, x10k, 100)
    info("CPU Forward 2")
    @time y2 = forw(n0, x10k, 100)
    assert(y1 == y2)
    info("CPU Forwback 1")
    @time forwback(n0, x10k, y10k, 100)
    info("CPU Forwback 2")
    @time forwback(n0, x10k, y10k, 100)

    device_reset(0)
    gw1 = CudaArray(w1)
    gb1 = CudaArray(b1)
    gw2 = CudaArray(w2)
    gb2 = CudaArray(b2)
    g0 = [relu(gw1,gb1), soft(gw2,gb2)]
    info("Running 10k gpu test with bias...")
    info("GPU Forward 1")
    @time y3 = forw(g0, x10k, 100)
    info("maxabsdiff=$(maximum(abs(y3-y2)))")
    info("GPU Forward 2")
    @time y4 = forw(g0, x10k, 100)
    assert(y4 == y3)
    info("GPU Forwback 1")
    @time forwback(g0, x10k, y10k, 100)
    info("GPU Forwback 2")
    @time forwback(g0, x10k, y10k, 100)
    info("done")
    return g0
#  end # devices


end # function test(data)

end # module Jnet
