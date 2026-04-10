using LinearAlgebra, OrdinaryDiffEq, StaticArrays
using Random, Statistics
using ComplexRegions, RationalFunctionApproximation
using GLMakie, DomainColoring
using ProgressMeter, Serialization
using #= MATLAB, =# DelimitedFiles
using Base.Iterators
using Base.Threads
using Base.Threads: Atomic, @threads
using Metal

rng = Random.seed!(1234)

#=
# Rössler vectorfield
function rhs(u, p, t)
    x, y, z, = u
    a, b, c, = p
    SA_F64[
        - y - z, 
        x + a * y, 
        b + z * (x - c)
    ]
end

function jac(u, p, t)
    x, y, z = u
    a, b, c = p
    SMatrix{3,3,Float64}(
        0,  -1,  -1,
        1,   a,   0,
        z,   0,  x-c
    )
end
=#
# Lorenz vectorfield 
function rhs(u, p, t)
    x, y, z = u
    ρ, σ, β = p
    SA_F32[
        σ * (y - x), 
        x * (ρ - z) - y, 
        x * y - β * z
    ]
end

function jac(u, p, t)
    x, y, z = u
    ρ, σ, β = p
    SA_F32[
        -σ    σ   0;
        ρ-z  -1  -x;
        y     x  -β
    ]
end

# vectorfield parameters we will search
n_iterations = 200
convex_param = range(0, 1, n_iterations)
#=
p_iterator = Float32[
    0.05 .* convex_param .+  0.25#=0.33=# .* (1 .- convex_param);; 
    0.4 * ones(n_iterations);;# 1.82 .* convex_param  .+  0.2 .* (1 .- convex_param);; 
    8.5 * ones(n_iterations);;# 9.75 .* convex_param  .+  5.7 .* (1 .- convex_param);; 
]
=#
p_iterator = Float32[
    23.5 .* convex_param .+ 24.5 .* (1 .- convex_param);;
    10 .* ones(n_iterations);; 
    8/3 .* ones(n_iterations);;
]

# observable 
g(x, σ=0f0) = x[1] - x[2] + 0.8f0x[3] - σ #exp(-norm(x)^2 / σ)#*sin(sum(x))  # something spicy
gσ(σ) = x -> g(x, σ)


# how many trajectories per parameter
n_trajectories = 30
plot_trajectories = false
#u0s = Ref(Diagonal(SA_F32[4, 4, 0.8])) .* rand(SVector{3,Float32}, n_trajectories) .- Ref(SA_F32[2, 2, 0.4])    # random initial conditions


# integration time 
N = 600         # number of delays
M = 2_000_000   # number of time steps

dt = 0.02f0     # time step
delay = 3dt
offset = Int32(delay ÷ dt)
tspan = (0, M*dt + (N+1)*delay)
integration_time = 0:dt:M*dt
dz = 1f0/length(integration_time)/n_trajectories  # multiplier for integral

autocorrelation = zeros(Float32, N)
autocorrelation_gpu = mtl(autocorrelation)
traj = mtl(rand(SVector{3,Float32}, M + (N+1)*offset + 1))

function inner_product_kernel(g, traj, corr, M, offset, dz)
    n = thread_position_in_grid().x
    integral = 0f0
    for i in 1:M
        u1 = traj[i]
        u2 = traj[i + (n-1)*offset]
        integral += g(u1)' * g(u2) * dz
    end
    corr[n] += integral
    return nothing
end

fourth_order_kernel(x) = 1 - x^4 * (-20abs(x)^3 + 70x^2 - 84abs(x) + 35)
Nkernel = round(Int, 3N/2)

#iteration = 40
# computation starts here 
# ---------------------------------------------------
prog = Progress(n_iterations, showspeed=true)
for iteration in 1:n_iterations

    # reset variables
    autocorrelation_gpu .= 0
    p = Tuple(p_iterator[iteration, :])


    if plot_trajectories
        # this plot gets all the trajectories
        fig3 = Figure()
        ax3 = Axis3(fig3[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "b", "c"), round.(p, digits=2))], ", "),
            azimuth=-0.5π
        )
    end


    #= # compute autocorrelations on GPU
    @showprogress for u0 in u0s
        prob = ODEProblem(ODEFunction(rhs, jac=jac), u0, tspan, p, saveat=dt)
        sol = solve(prob, Rosenbrock23(), dt=0.01f0, verbose=false, adaptive=false, maxiters=1f12)#, dtmin=0.01, force_dtmin=true)
        lines!(ax3, sol.u, color=:blue)
        
        Metal.synchronize()
        #traj = mtl(sol.u)
        copyto!(traj, sol.u)
        @metal threads=N inner_product_kernel(traj, autocorrelation_gpu, M, offset, dz)
    end
    
    Metal.synchronize()=#


    # set up a Channel of trajectories that we must compute
    unfinished_trajectories = Channel(n_trajectories)
    #foreach(x->put!(unfinished_trajectories, x), u0s)
    foreach(_->put!(unfinished_trajectories, rand(SVector{3,Float32}) + SA_F32[-1/2,-1/2,p[1]+p[2]-1/2]), 1:n_trajectories)
    close(unfinished_trajectories)
    finished_trajectories = Channel(n_trajectories)


    # compute trajectories - serial, but multiple can be prepared simultaneously
    cpu_tasks = [
        @spawn begin
            for u0 in unfinished_trajectories
                prob = ODEProblem(ODEFunction(rhs, jac=jac), u0, tspan, p, saveat=dt)
                sol = solve(prob, Rosenbrock23(), dt=0.01f0, verbose=false, adaptive=false, maxiters=1f12)#, dtmin=0.01, force_dtmin=true)
                put!(finished_trajectories, sol)
            end
        end
        for _ in 1:4
    ]


    # want to subtract mean to remove resonance at 1
    sol_sample = fetch(finished_trajectories)
    avg = mean(g, sol_sample.u)   
    g_zero_mean = gσ(avg)
    plot_trajectories  &&  lines!(ax3, plot_sol.u, color=:blue)


    # compute each autocorrelation on a GPU thread
    gpu_task = @spawn begin
        for sol in finished_trajectories
            copyto!(traj, sol.u)    # move data to GPU
            @metal threads=N inner_product_kernel(g_zero_mean, traj, autocorrelation_gpu, M, offset, dz)
            #global avg += mean(g, sol.u) / n_trajectories
            Metal.synchronize()     # compute autocorrelation
        end
    end


    foreach(fetch, cpu_tasks)       # wait for trajectories to finish calculating
    close(finished_trajectories)    # tell GPU task that CPU is done
    fetch(gpu_task)                 # wait for autocorrelations to finish calculating
    copyto!(autocorrelation, autocorrelation_gpu)   # move autocorrelation data to CPU


    # smoothing
    #autocorrelation .*= fourth_order_kernel.((0:N-1) ./ N)

    #zpts = exp.(im .* range(0, 2π, length=10_000))
    #meas(z) = autocorrelation' * z .^ (0:-1:-(N-1))
    #meas_pts = meas.(zpts)
    #four_meas = fft(meas_pts) / 10_000
    #meas_view = FFTView(four_meas)

    gnorm = autocorrelation[1]
    autocorrelation ./= gnorm

    N_symmetric = N ÷ 2 - 1

    G = [
        n ≥ m  ?  autocorrelation[n-m+1]  :  autocorrelation[m-n+1]'
        for m in -N_symmetric:N_symmetric, n in -N_symmetric:N_symmetric
    ]

    A = [
        n + 1 ≥ m  ?  autocorrelation[n-m+2]  :  autocorrelation[m-n]'
        for m in -N_symmetric:N_symmetric, n in -N_symmetric:N_symmetric
    ]

    H = sqrt(inv(Symmetric(G)))

    U, σ, V = try 
        svd(H * A' * H)
    catch err
        @warn "SVD died"
        serialize("svd_matrix.ser", H * A' * H)
        rethrow(err)
    end
    
    mpschur = schur(U * V')
    λ = mpschur.values
    V̂ = mpschur.vectors
    #= λ, V̂ = try
        eigen(V * U') 
    catch err
        @warn "eigen died"
        serialize("eig_matrix.ser", V * U')
        rethrow(err)
    end =#

    #perm = sortperm( λ, by=z -> angle(z) + (angle(z) ≥ 0 ? 0 : 2π) )
    #λ .= λ[perm]
    #V̂ .= V̂[:, perm]
    
    V = H * V̂
    
    #g_coeffs = zeros(2N_symmetric+1)
    #g_coeffs[N_symmetric+2] = 1
    #gnorm = G[N_symmetric+2,:]' * G[:,N_symmetric+2]#g_coeffs' * G * g_coeffs

    moments = exp.(-im .* (-N:N) .* angle.(λ)') * abs2.(V' * G[:,N_symmetric+2])# * g_coeffs)
    moments .*= fourth_order_kernel.( (-N:N) ./ N )
    

    # send autocorrelation data to REfit in matlab
    #writedlm("autocorrelation.txt", autocorrelation)
    writedlm("autocorrelation_real.txt", real.(moments))
    writedlm("autocorrelation_imag.txt", imag.(moments))

    resonance_run = chomp(read(`/Applications/MATLAB_R2025b.app/bin/matlab -batch "resonance_runner"`, String))
    poles_r = vec(readdlm("poles.txt", ComplexF64))
    poles_r[abs.(poles_r) .< 1e-20] .= 1e-20
    roots_r = vec(readdlm("roots.txt", ComplexF64))
    roots_r[abs.(roots_r) .< 1e-20] .= 1e-20


    # plot
    begin
        fig = Figure(size=(720,720))
        ax = Axis(fig[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "b", "c"), round.(p, digits=2))], ", "), 
            xlabel="Re(λ)", ylabel="Im(λ)"
        )
        lines!(ax, unitcircle, linewidth=1, color=:black)
        scatter!(ax, poles_r, markersize=10, color=:transparent, strokecolor=:black, strokewidth=3, marker=:circle)
        limits!(ax, -1.1, 1.1, -1.1, 1.1)#0.5, 1.1, -0.5, 0.5)
    end

    begin
        fig2 = Figure(size=(720,520))
        ax2 = Axis(fig2[1,1], 
            title=join(["$param = $val" for (param, val) in zip(("a", "b", "c"), round.(p, digits=2))], ", "), 
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
            title=join(["$param = $val" for (param, val) in zip(("a", "b", "c"), round.(p, digits=2))], ", "), 
            xlabel="t", ylabel="⟨𝒦ᵗg, g⟩"
        )
        #ms = plot!((0:N-1) .* delay, autocorrelation)
        ms = plot!(-N:N, real.(moments))
    end

    save("phaseplots/phaseplot_$iteration.png", fig)
    save("abs/abs_$iteration.png", fig2)
    save("power_spectra/power_$iteration.png", fig4)
    
    if plot_trajectories
        save("trajectories/traj_$iteration.png", fig3)
    end

    next!(prog, showvalues=[
        (:iteration, iteration), 
        (:observable_norm, sqrt(gnorm)), 
        (:matlab_output, resonance_run)
    ])

end


