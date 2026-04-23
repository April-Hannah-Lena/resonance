using ClassicalOrthogonalPolynomials, FastGaussQuadrature, LinearAlgebra
using ComplexRegions, RationalFunctionApproximation
using GLMakie, DomainColoring
using ProgressMeter, Serialization
using FLoops


#                defining the map
# -------------------------------------------------

function rk4(f, x, τ)
    τ½ = τ / 2

    k = f(x)
    dx = @. k / 6

    k = f(@. x + τ½ * k)
    dx = @. dx + k / 3

    k = f(@. x + τ½ * k)
    dx = @. dx + k / 3

    k = f(@. x + τ * k)
    dx = @. dx + k / 6

    return @. x + τ * dx
end

function rk4_flow_map(f, x, step_size=0.01, steps=20)
    for _ in 1:steps
        x = rk4(f, x, step_size)
    end
    return x
end

const _α, _ϵ, _ω = 0.25, 0.25, 2π

f(x, t)  = _ϵ * sin(_ω*t) * x^2 + (1 - 2_ϵ * sin(_ω*t)) * x
df(x, t) = 2_ϵ * sin(_ω*t) * x   + (1 - 2_ϵ * sin(_ω*t))

double_gyre(x, y, t) = (
    -π * _α * sin(π * f(x, t)) * cos(π * y),
     π * _α * cos(π * f(x, t)) * sin(π * y) * df(x, t)
)

# autonomize the ODE by adding a dimension
double_gyre((x, y, t)) = (double_gyre(x, y, t)..., 1)

# nonautonomous flow map: reduce back to 2 dims
function φ((x, y), t, τ, steps)
    (x, y, t) = rk4_flow_map(double_gyre, (x, y, t), τ, steps)
    return (x, y)
end

t₀, τ, steps = 0, 0.1, 20
t₁ = t₀ + τ * steps
φₜ₀ᵗ¹(z) = φ(z, t₀, τ, steps)

#=
const _γ = -1.7
f(z) = exp(-2*pi*im/3) * ( (abs(0.9z)^2 + _γ)*z + conj(0.9z)^2 / 2 )
fr((x, y)) = reim( f(x + y*im) )
=#


#                EDMD construction
# -------------------------------------------------

# we do cartesian product quadrature so M needs to be the square of an integer
M = round(Int, sqrt(20_000_000))^2
N = 100

#X = rand(M, 2)
#Y = permutedims(reinterpret(reshape, Float64, map(φₜ₀ᵗ¹, eachrow(X))), (2,1))

_X, _W = gausslegendre(round(Int, sqrt(M)))
#W = Diagonal(vec([w1*w2 for w1 in _W, w2 in _W]))

# the absolute worst way to do this but if it works don't fix it    <--  I fixed it
#X = permutedims(reinterpret(reshape, Float64, vec([(x1+1,x2/2+1/2) for x1 in _X, x2 in _X])), (2,1))
#Y = permutedims(reinterpret(reshape, Float64, map(φₜ₀ᵗ¹, eachrow(X))), (2,1))
#X = permutedims(reinterpret(reshape, Float64, vec([(x1,x2) for x1 in _X, x2 in _X])), (2,1))
#Y = permutedims(reinterpret(reshape, Float64, map(fr, eachrow(X))), (2,1))


legendre = Legendre()
function multivariate_index(n) # counts all pairs ℕ×ℕ 
    u = sqrt(8n + 1) - 1
    w = round(Int, u/2, RoundDown)
    t = (w^2 + w) / 2
    n1 = n - t
    n2 = w - n1
    return Int(n1+1), Int(n2+1)
end
indices = multivariate_index.(0:N-1)
max_index = maximum(reinterpret(Int64, indices))


#=
space = Legendre(0..2) ⊗ Legendre(0..1)
#space = Legendre() ⊗ Legendre()
_Ψ = [Fun(space, [zeros(n)..., sqrt((2n+1)/2)]/(isodd(n) ? sqrt(2) : 1)) for n in 0:N1-1]
_Φ = [Fun(space, [zeros(n)..., sqrt((2n+1)/2)]/(isodd(n) ? sqrt(2) : 1)) for n in 0:N2-1]


Ψ((x, y)) = map(f -> f(x,y), _Ψ)
Φ((x, y)) = map(f -> f(x,y), _Φ)

ΨX = zeros(M, N1)
ΦY = zeros(M, N2)
prog = Progress(M)
@threads for m in axes(X,1)
    x, y = X[m,:], Y[m,:]
    ΨX[m,:] .= Ψ(x)
    ΦY[m,:] .= Φ(y)
    next!(prog)
end

G = ΨX' * W * ΨX
A = ΨX' * W * ΦY
=#

prog = Progress(length(_X)^2)
@floop for (_x1, w1) in zip(_X, _W), (_x2, w2) in zip(_X, _W)
    
    @init ξx1 = zeros(max_index)  # these @init do nothing outside of
    @init ξx2 = zeros(max_index)  # making it run a little faster
    @init ξy1 = zeros(max_index)
    @init ξy2 = zeros(max_index)
    @init Ξx = zeros(N)
    @init Ξy = zeros(N)
    
    x1, x2  =  _x1 + 1,  _x2/2 + 1/2
    y1, y2 = φₜ₀ᵗ¹((x1,x2))
    _y1, _y2  =  y1 - 1,  2y2 - 1

    @. ξx1 = legendre[_x1, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    @. ξx2 = legendre[_x2, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    @. ξy1 = legendre[_y1, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    @. ξy2 = legendre[_y2, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    @. Ξx = [ξx1[i]*ξx2[j] for (i,j) in indices]
    @. Ξy = [ξy1[i]*ξy2[j] for (i,j) in indices]

    @reduce( 
        G = zeros(N,N) + (w1*w2 .* Ξx*Ξx'),  
        A = zeros(N,N) + (w1*w2 .* Ξx*Ξy')
    )
    #G += w1*w2 .* Ξx*Ξx'
    #A += w1*w2 .* Ξx*Ξy'

    next!(prog)
end



#                resonances
# -------------------------------------------------

# pick some random observables
e0_coeffs = zeros(ComplexF64, N)
em_coeffs = zeros(ComplexF64, N)
e0_coeffs[8] = 1
em_coeffs[4] = 1

#=
is_on_diag(ind::CartesianIndex) = ind.I[1] == ind.I[2]
is_on_diag(mat::Matrix) = is_on_diag.(CartesianIndices(mat))
function I_nonsquare(A)
    I = similar(A)
    I[is_on_diag(I)] .= 1
    I[.!is_on_diag(I)] .= 0
    return I
end
=#

function resolvent(λ⁻¹, A=A', u=e0_coeffs, v=em_coeffs)     # ⟨ (𝒦 - λ⁻¹)⁻¹ u, v ⟩
    w = (A  -  λ⁻¹ * I) \ u
    return w'v      # this works because orthonormal basis
end


ϵ = 1e-14
r = approximate(resolvent, (1-ϵ)*unitdisk)

#fig = Figure(size=(1200,1200))
#ax = Axis(fig[1,1])
begin
domaincolor(z -> r(1/z), 1.1, abs=true)
lines!(unitcircle, color=:white, linewidth=2)
scatter!(1 ./ nodes(r), markersize=10, color=:green, marker=:x)
scatter!((x->abs(x)>1e6 ? 1e6+0im : x).(1 ./ roots(r)), markersize=8, color=:transparent, strokecolor=:white, strokewidth=2, marker=:circle)
scatter!(1 ./ poles(r), markersize=8, color=:transparent, strokecolor=:black, strokewidth=2, marker=:circle)
#scatter!(λ_true, markersize=6, color=:blue, marker=:+)
#limits!(0.8, 1.1, -0.2, 0.2)
limits!(-1.1, 1.1, -1.1, 1.1)
current_figure()
end

#=
serialize("matrix_representation.ser", A)
serialize("e0_coeffs.ser", e0_coeffs)
serialize("em_coeffs.ser", em_coeffs)
=#

save("gyre_phaseplot.png", current_figure(), px_per_unit=3.0)

#                resonant modes
# -------------------------------------------------

# take the pole closest to 
λ = 0.99

function mode_coeffs(k, λ, ϵ=ϵ, A=A, u=e0_coeffs)
    v = zeros(ComplexF64, N)
    v[k] = 1
    r = approximate(z -> resolvent(z, A, u, v), (1-ϵ)*unitdisk)


    _poles, res = residues(r)
    poles = 1 ./ _poles

    n = argmin(abs.(poles .- λ))
    return res[n]
end

coeffs = mode_coeffs.(1:N, λ)

x1s = 0:0.01:2
x2s = 0:0.01:1
mode = zeros(ComplexF64, length(x1s), length(x2s))
for k in CartesianIndices(mode)
    k1, k2 = Tuple(k)
    _x1, _x2 = x1s[k1] - 1,  2x2s[k2] - 1
    ξx1 = @. legendre[_x1, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    ξx2 = @. legendre[_x2, 1:max_index] * sqrt(2 * (0:max_index-1) + 1) / sqrt(2)
    Ξ = [ξx1[i]*ξx2[j] for (i,j) in indices]
    mode[k1,k2] = coeffs' * Ξ
end

contourf(x1s, x2s, real.(mode))

save("gyre_resonance.png", current_figure(), px_per_unit=3.0)
