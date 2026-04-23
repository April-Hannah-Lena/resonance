using LinearAlgebra, OrdinaryDiffEq, StaticArrays
using Random, Statistics
using FastGaussQuadrature
using ComplexRegions, RationalFunctionApproximation
using GLMakie, DomainColoring, LaTeXStrings
using GLMakie.Makie: pseudolog10
using ProgressMeter, Serialization
using #= MATLAB, =# DelimitedFiles
using Base.Iterators
using Base.Threads
import ThreadsX as tx
using Base.Threads: Atomic, @threads


rng = Random.seed!(1234)

# Van der Pol vectorfield 
function rhs(u, p, t)
    x, y = u
    SA_F32[
        p*x - y - x*(x^2 + y^2),
        x + p*y - y*(x^2 + y^2)
    ]
end


# vectorfield parameters we will search
n_iterations = 20
p_iterator = range(0.2f0, -0.2f0, length=n_iterations)

# observable 
g(x, σ=0f0) = x[1] - x[2] - σ #exp(-norm(x)^2 / σ)#*sin(sum(x))  # something spicy
gσ(σ) = x -> g(x, σ)


# how many trajectories per parameter
n_trajectories = 30_000
plot_trajectories = false


_nodes, _weights = gausslegendre(round(Int, sqrt(n_trajectories), RoundDown))
u0s = [SA_F32[x,y] for x in _nodes, y in _nodes]
weights = [wx*wy for wx in _weights, wy in _weights]


# integration time 
N = 300         # number of delays
M = 1   # number of time steps

dt = π/6f0     # time step
delay = 100dt
offset = Int32(delay ÷ dt)
tspan = (0, M*dt + (N+1)*delay)
integration_time = 0:dt:M*dt
#dz = 1f0/length(integration_time)/n_trajectories  # multiplier for integral


autocorrelation = zeros(Float32, N)

second_order_kernel(x) = (1 + cos(π*x)) / 2
fourth_order_kernel(x) = 1 - x^4 * (-20abs(x)^3 + 70x^2 - 84abs(x) + 35)
N_smoothing = N
residue_tolerance = 1e-5


#iteration = 40
# computation starts here 
# ---------------------------------------------------
prog = Progress(n_iterations, showspeed=true)
for iteration in 1:n_iterations

    # reset variables
    autocorrelation .= 0
    p = p_iterator[iteration]


    if plot_trajectories
        # this plot gets all the trajectories
        fig3 = Figure()
        ax3 = Axis3(fig3[1,1], 
            title="p = $(round(p, digits=4))",
            azimuth=-0.5π
        )
    end


    # compute autocorrelations 
    autocorrelation .= tx.sum(zip(u0s, weights)) do (u0, w)
        prob = ODEProblem(ODEFunction(rhs), u0, tspan, p, saveat=dt)
        sol = solve(prob, Tsit5(), dt=0.25f0, verbose=false, adaptive=false, maxiters=1f12)#, dtmin=0.01, force_dtmin=true)
        SVector{N,Float32}(g(sol(0))' * g(sol(n*dt)) * w for n in 0:N-1)
    end


    gnorm = autocorrelation[1]
    autocorrelation ./= gnorm

    # smoothing
    autocorrelation .*= second_order_kernel.((0:N-1) ./ N_smoothing)

    zpts = exp.(im .* range(0, 2π, length=10_000))
    meas(z) = autocorrelation' * z .^ (0:-1:-(N-1))
    meas_pts = meas.(zpts)
    #four_meas = fft(meas_pts) / 10_000
    #meas_view = FFTView(four_meas)


    # send autocorrelation data to REfit in matlab
    writedlm("autocorrelation.txt", autocorrelation)
    #writedlm("autocorrelation_real.txt", real.(moments))
    #writedlm("autocorrelation_imag.txt", imag.(moments))

    resonance_run = chomp(read(`/Applications/MATLAB_R2025b.app/bin/matlab -batch "resonance_runner"`, String))
    poles_r = vec(readdlm("poles.txt", ComplexF64))
    roots_r = vec(readdlm("roots.txt", ComplexF64))
    residues_r = vec(readdlm("residues.txt", ComplexF64))

    residues_r = residues_r[abs.(poles_r) .≤ 1]
    poles_r = poles_r[abs.(poles_r) .≤ 1]
    roots_r = roots_r[abs.(roots_r) .≤ 1]

    perm = sortperm(residues_r, by=abs)
    n_bad_poles = sum(abs.(residues_r) .< residue_tolerance)
    n_bad_poles = min(n_bad_poles, length(poles_r)-1)   # give us at least one thing to plot

    poles_r = poles_r[perm[n_bad_poles+1:end]]
    residues_r = residues_r[perm[n_bad_poles+1:end]]


    # plot
    begin
        fig = Figure(size=(720,720))
        ax = Axis(fig[1,1], 
            title="p = $(round(p, digits=4))", 
            xlabel="Re(λ)", ylabel="Im(λ)"
        )
        lines!(ax, unitcircle, linewidth=1, color=:black)
        ms = scatter!(
            ax, poles_r, 
            markersize=10, 
            color=abs.(residues_r), 
            strokewidth=3, 
            marker=:circle, 
            #colorscale=pseudolog10, 
            #colorrange=(-0.05, 0.351)
        )
        limits!(ax, -1.1, 1.1, -1.1, 1.1)#0.5, 1.1, -0.5, 0.5)
        cb = Colorbar(fig[1,2], ms)
        #cb.ticks = (collect(-0.05:0.1:0.35), [L"10^{%$(k)}" for k in -0.05:0.1:0.35])
    end

    begin
        fig2 = Figure(size=(720,520))
        ax2 = Axis(fig2[1,1], 
            title="p = $(round(p, digits=4))", 
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
            title="p = $(round(p, digits=4))", 
            xlabel="t", ylabel="⟨𝒦ᵗg, g⟩"
        )
        ms = plot!((0:N-1) .* delay, autocorrelation)
        #ms = plot!(-N:N, real.(moments))
    end

    begin
        fig5 = Figure(size=(720,520))
        ax5 = Axis(fig5[1,1],
            title="p = $(round(p, digits=4))", 
            xlabel="θ", ylabel="power spectral density", 
            xticks=round.(-π:π/4:π, digits=3), 
            yscale=pseudolog10
        )
        ms = plot!(angle.(zpts), real.(meas_pts), label="real")
        ms = plot!(angle.(zpts), imag.(meas_pts), label="imaginary")
        #ms = plot!(-N:N, real.(moments))
    end

    save("phaseplots/phaseplot_$iteration.png", fig)
    save("abs/abs_$iteration.png", fig2)
    save("autocorrelations/auto_$iteration.png", fig4)
    save("power_spectra/power_$iteration.png", fig5)
    
    if plot_trajectories
        save("trajectories/traj_$iteration.png", fig3)
    end

    next!(prog, showvalues=[
        (:iteration, iteration), 
        (:observable_norm, sqrt(gnorm)), 
        (:matlab_output, resonance_run)
    ])

end


