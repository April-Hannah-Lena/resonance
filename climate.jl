using DelimitedFiles, FileIO
using Statistics
using ComplexRegions
using ProgressMeter
using GLMakie


# I think these files refer to years? Not sure...
years = [69, 77, 81, 82, 83]

types = Dict(
    "deep" => 17:20,
    "middle" => 6:9,
    "surface" => 2:5
)


# observable
g(x) = x[1] - x[2] + x[3] - x[4]


N = 100     # number of delays
delay = 5   # delay length


prog = Progress(length(years)*length(types))
for year in years, (type, columns) in types


    # get data
    data = readdlm("veros_data/data_Veros_f$(year)_red24.csv", ',', Float64)'

    # for whatever reason only these variables are interesting
    data = data[1001:end, columns] 
    data .= (data .- mean(data, dims=1)) ./ std(data, dims=1)


    begin
        fig1 = Figure()
        lines!(Axis(fig1[1,1]), data[:,1])
        lines!(Axis(fig1[1,2]), data[:,2])
        lines!(Axis(fig1[2,1]), data[:,3])
        lines!(Axis(fig1[2,2]), data[:,4])
        Label(fig1[0, :], "$type water data", fontsize=14, font=:bold, tellwidth=false)
        #current_figure()
    end


    observable = g.(eachrow(data[1:end-N*delay, :]))

    autocorrelations = map(0:N-1) do n
        delay_data = data[n*delay+1:end-(N-n)*delay, :]
        delay_observable = g.(eachrow(delay_data))
        observable' * delay_observable
    end


    writedlm("autocorrelation.txt", autocorrelations)    
    run(`/Applications/MATLAB_R2025b.app/bin/matlab -batch "resonance_runner"`)
    poles_r = vec(readdlm("poles.txt", ComplexF64))
    poles_r[abs.(poles_r) .< 1e-20] .= 1e-20
    roots_r = vec(readdlm("roots.txt", ComplexF64))
    roots_r[abs.(roots_r) .< 1e-20] .= 1e-20


    begin
        fig2 = Figure(size=(720,720))
        ax = Axis(fig2[1,1], xlabel="Re(λ)", ylabel="Im(λ)", title="$type water resonances")
        lines!(ax, unitcircle, linewidth=1, color=:black)
        scatter!(ax, poles_r, markersize=10, color=:transparent, strokecolor=:black, strokewidth=3, marker=:circle)
        limits!(ax, -1.1, 1.1, -1.1, 1.1)
        #current_figure()
    end

    begin
        fig3 = Figure(size=(720,520))
        ax4 = Axis(fig3[1,1], xlabel="t", ylabel="⟨𝒦ᵗg, g⟩", title="$type water power spectrum")
        ms = plot!((0:N-1) .* delay, autocorrelations)
        #current_figure()
    end


    save("climate_plots/$type/data/$year.png", fig1)
    save("climate_plots/$type/phaseplot/$year.png", fig2)
    save("climate_plots/$type/power_spectrum/$year.png", fig3)

    next!(prog)

end


type = "deep"#["deep", "middle", "surface"]
fig = Figure()

for (year_index, year) in enumerate(years), 
        (plot_index, plottype) in enumerate(["data", "power_spectrum", "phaseplot"])

    ax = Axis(fig[plot_index, year_index], xlabel="$year", aspect=DataAspect(), #=xlabel="$type"=#)
    img = load("climate_plots/$type/$plottype/$year.png")
    image!(ax, rotr90(img))
    hidedecorations!(ax, label=false)

end


plottype = "phaseplot"#["data", "power_spectrum", "phaseplot"] 
fig = Figure()

for (year_index, year) in enumerate(years), 
        (type_index, (type, columns)) in enumerate(types)

    ax = Axis(fig[type_index, year_index], xlabel="$year", aspect=DataAspect(), #=xlabel="$type"=#)
    img = load("climate_plots/$type/$plottype/$year.png")
    image!(ax, rotr90(img))
    hidedecorations!(ax, label=false)

end
