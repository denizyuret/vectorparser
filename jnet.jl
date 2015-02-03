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
    initforw(l, x)
    gemm!('N', 'N', one(t), l.w, l.x, zero(t), l.y)  # y=w*x
    if isdefined(l, :b) 
        # broadcast!(+, l.y, l.y, l.b)
        ger!(one(t), l.b, l.xones, l.y)	 # y=y+b: add b to each column of y
    end
    if isdefined(l, :fforw) 
        l.fforw(l.y)  # y=fforw(y)
    end
    return l.y
end

function back{t}(l::Layer{t}, dy::Mat{t}, dx=true)
    initback(l, dy, dx)
    if (isdefined(l, :fback)) 
        l.fback(dy, l.y)  # this will overwrite dy and l.y!
    end
    gemm!('N', 'T', one(t), dy, l.x, zero(t), l.dw)  # dw=dy*x'
    if (isdefined(l, :b)) 
        # sum!(l.db, dy)
        gemv!('N', one(t), dy, l.xones, zero(t), l.db)  # db=sum(dy,2)
    end
    if (dx)  # dx is optional because it is expensive and unnecessary for input layer
        gemm!('T', 'N', one(t), l.w, dy, zero(t), l.dx)  # dx=w'*dy
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
    for i=1:length(net)
        x = forw(net[i], x)
    end
    return x
end

function back{t}(net::Net{t}, dy::Mat{t})
    for i=length(net):-1:2
        dy = back(net[i], dy)
    end
    # No need to compute the last dx
    back(net[1], dy, false)
end


### Batch processing:

function forw{t}(net::Net{t}, x::Mat{t}, batch::Int)
    xcols = size(x, 2)
    y = zeros(t, size(net[end].w, 1), xcols)
    info("forw:Alloc(y)=$(mysizeof(y))")
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        # copy!(sub(y,:,b:e), forw(net, sub(x,:,b:e)))
        # y[:,b:e] = forw(net, sub(x,:,b:e))
        # y[:,b:e] = to_host(forw(net, sub(x,:,b:e)))
        foo = forw(net, sub(x,:,b:e))
        # display(foo)
        y[:,b:e] = to_host(foo)
    end
    return y
end

function forwback{t}(net::Net{t}, x::Mat{t}, labels::Vec{t}, batch::Int)
    xcols = size(x, 2)
    y = zeros(t, size(net[end].w, 1), xcols)
    info("forwback:Alloc(y)=$(mysizeof(y))")
    for i=1:length(labels)
        y[labels[i],i] = 1
    end
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        # info("$((b,e))")
        forw(net, sub(x,:,b:e))
        back(net, sub(y,:,b:e))
    end
end


### Memory management for forw and back
# This code is ugly because:
# 1. Need to free CudaArrays rather than relying on gc()
# 2. With getindex, ones, etc. missing from CUDArt, not possible to write generic code

function initforw{t}(l::Layer{t}, x::Mat{t})
    if (!isdefined(l, :w))
        error("l.w not defined")
    end
    l.x = x
    rows = size(l.w,1)
    cols = size(l.x,2)
    if (!isdefined(l, :y) || (size(l.y) != (rows, cols)))
        if (isdefined(l, :y)) free(l.y); end
        l.y = similar(l.w, rows, cols)
        info("initforw:Alloc(y)=$(mysizeof(l.y))")
    end
    if (isdefined(l, :b) && (!isdefined(l, :xones) || (length(l.xones) != cols)))
        if (isdefined(l, :xones)) free(l.xones); end
        l.xones = ones(l.w, cols)
        info("initforw:Alloc(xones)=$(mysizeof(l.xones))")
    end
end

function initback{t}(l::Layer{t}, dy::Mat{t}, dx::Bool)
    if (!isdefined(l, :dw)) 
        l.dw = zeros(l.w)
        info("initback:Alloc(dw)=$(mysizeof(l.dw))")
    end
    if (isdefined(l, :b) && !isdefined(l, :db))
        l.db = zeros(l.b)
        info("initback:Alloc(db)=$(mysizeof(l.db))")
    end
    if (dx && (!isdefined(l, :dx) || (size(l.dx) != size(l.x))))
        if (isdefined(l, :dx)) free(l.dx); end
        l.dx = zeros(l.x)
        info("initback:Alloc(dx)=$(mysizeof(l.dx))")
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
    info("CPU Forwback 3")
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
    return n0
#  end # devices


end # function test(data)

end # module Jnet
