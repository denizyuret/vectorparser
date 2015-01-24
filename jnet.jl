module Jnet

using NumericExtensions
using Base.LinAlg.BLAS
blas_set_num_threads(12)

typealias Mat{t} AbstractMatrix{t}
typealias Vec{t} AbstractVector{t}

type Layer{t}
    w::Mat{t}
    b::Vec{t}
    fforw::Function
    fback::Function
    dw::Mat{t}
    db::Vec{t}
    dx::Mat{t}
    x::Mat{t}
    y::Mat{t}
    Layer()=new()
    Layer(w,b,f,g)=new(w,b,f,g)
end

function forw{t}(l::Layer{t}, x::Mat{t})
    initforw(l, x)
    gemm!('N', 'N', one(t), l.w, l.x, zero(t), l.y)
    if isdefined(l, :b) 
        broadcast!(+, l.y, l.y, l.b)
    end
    if isdefined(l, :fforw) 
        l.fforw(l.y)
    end
    return l.y
end

function back{t}(l::Layer{t}, dy::Mat{t}, dx=true)
    initback(l, dy, dx)
    if (isdefined(l, :fback)) 
        l.fback(dy, l.y)  # this will overwrite dy
    end
    gemm!('N', 'T', one(t), dy, l.x, zero(t), l.dw)
    if (isdefined(l, :b)) 
        sum!(l.db, dy)
    end
    if (dx) 
        gemm!('T', 'N', one(t), l.w, dy, zero(t), l.dx)
        return l.dx
    end
end

function initforw{t}(l::Layer{t}, x::Mat{t})
    l.x = x
    if (!isdefined(l, :w))
        error("l.w not defined")
    end
    rows = size(l.w,1)
    cols = size(l.x,2)
    if (!isdefined(l, :y) || size(l.y) != (rows, cols))
        l.y = zeros(t, rows, cols)
    end
end

function initback{t}(l::Layer{t}, dy::Mat{t}, dx::Bool)
    if (!isdefined(l, :dw)) 
        l.dw = 0 * l.w 
    end
    if (isdefined(l, :b) && !isdefined(l, :db))
        l.db = 0 * l.b
    end
    if (dx && !isdefined(l, :dx))
        l.dx = 0 * l.x
    end
end


relu{t}(w::Mat{t},b::Vec{t}) = Layer{t}(w, b, reluforw, reluback)
soft{t}(w::Mat{t},b::Vec{t}) = Layer{t}(w, b, softforw, softback)

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
    # we do softmax here instead of forw
    softmax!(y,y,1)
    # dy is a 0-1 matrix of correct answers
    # will overwrite it with the gradient
    ncols = size(y,2)
    for i=1:length(dy)
        dy[i] = (y[i] - dy[i]) / ncols
    end
end


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
    # No need to compute the last dy
    back(net[1], dy, false)
end

function forw{t}(net::Net{t}, x::Mat{t}, batch::Int)
    xcols = size(x, 2)
    y = zeros(t, size(net[end].w, 1), xcols)
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        y[:,b:e] = forw(net, sub(x,:,b:e))
    end
    return y
end

function forwback{t}(net::Net{t}, x::Mat{t}, labels::Vec{t}, batch::Int)
    xcols = size(x, 2)
    y = zeros(t, size(net[end].w, 1), xcols)
    for i=1:length(labels)
        y[labels[i],i] = 1
    end
    for b=1:batch:xcols
        e = b + batch - 1
        if (e > xcols) e = xcols; end
        forw(net, sub(x,:,b:e))
        back(net, sub(y,:,b:e))
    end
end


using MAT

function load()
    file = matopen("dev.mat")
    dev = read(file, "dev")
    close(file)
    return dev
end

function test(dev)
    # info("Loading...")
    # dev = load()
    info("Initializing...")
    n01 = relu(dev["w1"][:,2:end], dev["w1"][:,1])
    n02 = soft(dev["w2"][:,2:end], dev["w2"][:,1])
    n0 = [n01,n02]
    x10k = dev["x"][:,1:10000]
    y10k = dev["y"][1:10000]
    info("Running 10k test with bias...")
    info("Forward 1")
    @time forw(n0, x10k, 100)
    info("Forward 2")
    @time forw(n0, x10k, 100)
    info("Forwback 1")
    @time forwback(n0, x10k, y10k, 100)
    info("Forwback 2")
    @time forwback(n0, x10k, y10k, 100)
    info("done")
    return n0
end

end
