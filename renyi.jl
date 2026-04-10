using ApproxFun, LinearAlgebra, FastGaussQuadrature
using ComplexRegions, RationalFunctionApproximation
using GLMakie, DomainColoring
using ProgressMeter, Serialization
using Base: Threads
using Base.Threads: @spawn, @threads


legendre_space = Legendre(0..1)     # leg. poly.s transplanted to 0..1

β = 2.0
f(x, β) = β*x % 1
f(β) = x -> f(x, β)

λ_true = β .^ (-1:-1:-10)

M = 100_000
N = 50

dictionary = [
    √(2n+1)*Fun(legendre_space, [zeros(n); 1]) 
    for n in 0:N-1
]

xs, ws = gausslegendre(M)
xs .= xs ./ 2  .+  1/2
ws ./= 2

e0(z) = dictionary[2](z)
em(z) = dictionary[2](z)


# ---------------------------------------------------------
#=
A = [    # ⟨ 𝒦ψᵢ, ϕⱼ ⟩,  i = 1, ..., N,  j = 1, ..., N
    sum( (ψ ∘ f_complex)(z)' * ϕ(z) for z in circ ) * dθ
    for ϕ in dictionary, ψ in dictionary
] 

e0_coeffs = [   # ⟨ e0, ϕⱼ ⟩,  j = 1, ..., N
    sum( ϕ(z)' * e0(z) for z in circ ) * dθ
    for ϕ in dictionary
]
em_coeffs = [   # ⟨ em, ϕⱼ ⟩,  j = 1, ..., N
    sum( ϕ(z)' * e0(z) for z in circ ) * dθ
    for ϕ in dictionary
]
=#
# ---------------------------------------------------------

A = zeros(ComplexF64, length(dictionary),length(dictionary))

prog = Progress(length(A))
@threads for index in CartesianIndices(A)
    i,j = Tuple(index)
    ϕ, ψ = dictionary[i], dictionary[j]
    A[index] = sum( (ψ ∘ f(β))(z)' * ϕ(z) * w for (z,w) in zip(xs,ws) )
    next!(prog)
end

e0_coeffs = zeros(ComplexF64, length(dictionary))
em_coeffs = zeros(ComplexF64, length(dictionary))

prog = Progress(length(e0_coeffs))
@threads for index in eachindex(e0_coeffs)
    ϕ = dictionary[index]
    e0_coeffs[index] = sum( ϕ(z)' * e0(z) * w for (z,w) in zip(xs,ws) )
    em_coeffs[index] = sum( ϕ(z)' * em(z) * w for (z,w) in zip(xs,ws) )
    next!(prog)
end
 
# ---------------------------------------------------------

function resolvent(λ⁻¹, A=A, u=e0_coeffs, v=em_coeffs)     # ⟨ (𝒦 - λ⁻¹)⁻¹ u, v ⟩
    w = (A  -  λ⁻¹ * I) \ u
    return w'v      # this works bc fourier is orthonormal basis
end


ϵ = 1e-14
r = approximate(resolvent, (1-ϵ)*unitdisk)

#fig = Figure(size=(1200,1200))
#ax = Axis(fig[1,1])
begin
domaincolor(z -> r(1/z), 1.1, abs=true)
lines!(unitcircle, color=:white, linewidth=2)
scatter!(1 ./ poles(r), markersize=8, color=:transparent, strokecolor=:black, strokewidth=2, marker=:circle)
scatter!(ComplexF64.(λ_true), markersize=6, color=:blue, marker=:+)
scatter!(1 ./ nodes(r), markersize=6, color=:green, marker=:x)
limits!(-0.5, 1.1, -0.8, 0.8)
current_figure()
end

# ---------------------------------------------------------
#=
serialize("matrix_representation.ser", A)
serialize("e0_coeffs.ser", e0_coeffs)
serialize("em_coeffs.ser", em_coeffs)
=#