using ApproxFun, LinearAlgebra
using ComplexRegions, RationalFunctionApproximation
using GLMakie, DomainColoring
using ProgressMeter, Serialization
using Base: Threads
using Base.Threads: @spawn, @threads


μ = -0.33*exp(2π*im/200)
f_complex(z, μ=μ) = (z - μ)^2 / (1 - μ'z)^2
f(θ, μ=μ) = angle( f_complex( exp(2π*im*θ) ) ) / (2π)  +  1/2

kappa = 0.9311625603 + 0.0974629293*im
λ_true = vec( [kappa kappa'] .^ (0:50) )

basis(z, n) = z^n
basis(n) = z -> z^n

M = 10_000_000
N = 20

dictionary = basis.(-N:N)

dθ = 1/M
θs = dθ:dθ:1
circ = ( exp(2π*im*θ) for θ in θs )

e0(z) = z^5
em(z) = z^3

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
    A[index] = sum( (ψ ∘ f_complex)(z)' * ϕ(z) * dθ for z in circ )
    next!(prog)
end

e0_coeffs = zeros(ComplexF64, length(dictionary))
em_coeffs = zeros(ComplexF64, length(dictionary))

prog = Progress(length(e0_coeffs))
@threads for index in eachindex(e0_coeffs)
    ϕ = dictionary[index]
    e0_coeffs[index] = sum( ϕ(z)' * e0(z) * dθ for z in circ )
    em_coeffs[index] = sum( ϕ(z)' * em(z) * dθ for z in circ )
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
scatter!(λ_true, markersize=6, color=:blue, marker=:+)
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