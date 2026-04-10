using LinearAlgebra, StaticArrays, Random
using ComplexRegions
using GLMakie, DomainColoring
using ProgressMeter, Serialization
using #= MATLAB, =# DelimitedFiles
using Base.Iterators
using Base.Threads
using Base.Threads: @threads
using Metal

rng = Random.seed!(1234)


# 3D Hénon map
α = 0.1f0
σ = 0.1f0
δ = 0.7f0
function henon((x,y,z), p)
    α, σ, δ = p
    ( δ*z + α - σ*y + x^2,  x,  y )
end


# parameters we will search
n_iterations = 200
convex_param = range(0, 1, n_iterations)
p_iterator = Float32[
    @. -0.28960 * convex_param  +  0.07064 * (1 - convex_param);;
    0.62724 * ones(n_iterations);; 
    0.7 * ones(n_iterations);;
]

# observable 
g(x, σ=200f0) = exp(-norm(x)^2 / σ)# x[1] - x[2] + 0.8f0x[3]#*sin(sum(x))  # something spicy


# how many trajectories per parameter
n_trajectories = 300
plot_trajectories = true


# The cube | (x,y,z) |_∞ ≤ κ contains all bounded orbits
κ = (abs(σ) + δ + 1 + sqrt((abs(σ) + δ + 1)^2 + 4abs(α))) / 2

u0s = rand(SVector{3,Float32}, n_trajectories) 
u0s .-= Ref(SA_F32[1/2,1/2,1/2])
u0s .*= κ


# integration time 
N = 500         # number of delays
M = 20_000   # number of time steps

delay = 3
dz = 1f0/(M*n_trajectories)  # multiplier for integral

autocorrelation = zeros(Float32, N)
autocorrelation_gpu = mtl(autocorrelation)
traj = mtl(rand(SVector{3,Float32}, M + (N+1)*delay + 1))

function inner_product_kernel(traj, corr, M, offset, dz)
    n = thread_position_in_grid().x
    integral = 0f0
    for i in 1:M
        u1 = traj[i]
        u2 = traj[i + (n-1)*offset]
        inner = g(u1)' * g(u2) * dz
        integral += inner
        #isnan(integral)  &&  throw(ErrorException("$inner"))
    end
    corr[n] += integral
    return nothing
end


iteration = 1
# computation starts here 
# ---------------------------------------------------
#@showprogress for iteration in 1:n_iterations

    autocorrelation_gpu .= 0
    p = Tuple(p_iterator[iteration, :])


    if plot_trajectories
        # this plot gets all the trajectories
        fig3 = Figure()
        ax3 = Axis3(fig3[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "σ", "δ"), round.(p, digits=3))], ", "),
            azimuth=-0.5π
        )
    end


    # compute autocorrelations on GPU
    @showprogress for u0 in u0s
        sol = Vector{SVector{3,Float32}}(undef, M + (N+1)*delay + 1)
        sol[1] = u0
        for n in 1:length(sol)-1
            sol[n+1] = henon(sol[n], p)
        end
        
        copyto!(traj, sol)
        if any(x -> any(>(10), x), sol)
            @warn "why are you diverging"
            continue
        end

        @metal threads=N inner_product_kernel(traj, autocorrelation_gpu, M, delay, dz)
        Metal.synchronize()
    end
    

    copyto!(autocorrelation, autocorrelation_gpu)


    # send autocorrelation data to REfit in matlab
    writedlm("autocorrelation.txt", autocorrelation)
    run(`/Applications/MATLAB_R2025b.app/bin/matlab -batch "resonance_runner"`)
    poles_r = vec(readdlm("poles.txt", ComplexF64))
    poles_r[abs.(poles_r) .< 1e-20] .= 1e-20
    roots_r = vec(readdlm("roots.txt", ComplexF64))
    roots_r[abs.(roots_r) .< 1e-20] .= 1e-20


    # plot
    begin
        fig = Figure(size=(720,720))
        ax = Axis(fig[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "σ", "δ"), round.(p, digits=3))], ", "), 
            xlabel="Re(λ)", ylabel="Im(λ)"
        )
        lines!(ax, unitcircle, linewidth=1, color=:black)
        scatter!(ax, poles_r, markersize=10, color=:transparent, strokecolor=:black, strokewidth=3, marker=:circle)
        limits!(ax, -1.1, 1.1, -1.1, 1.1)#0.5, 1.1, -0.5, 0.5)
    end

    begin
        fig2 = Figure(size=(720,520))
        ax2 = Axis(fig2[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "σ", "δ"), round.(p, digits=3))], ", "), 
            xlabel="Arg(λ)", ylabel="|λ|",
            xticks=round.(-π:π/4:π, digits=3),
            yscale=log10
        )
        ms = scatter!(ax2, angle.(poles_r), abs.(poles_r), markersize=12, marker=:x, label="poles")
        limits!(ax2, -π-0.02, π+0.02, 0.5, 1.05)
        axislegend(ax2, position=:rb)
    end

    begin
        fig4 = Figure(size=(720,520))
        ax4 = Axis(fig4[1,1],
            title=join(["$param = $val" for (param, val) in zip(("a", "σ", "δ"), round.(p, digits=3))], ", "), 
            xlabel="t", ylabel="⟨𝒦ᵗg, g⟩"
        )
        ms = plot!((0:N-1) .* delay, autocorrelation)
    end

    save("phaseplots/phaseplot_$iteration.png", fig)
    save("abs/abs_$iteration.png", fig2)
    save("power_spectra/power_$iteration.png", fig4)
    
    if plot_trajectories
        save("trajectories/traj_$iteration.png", fig3)
    end

end


