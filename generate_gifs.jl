using Plots, Images

n_iterations = 150
phaseplots = @animate for it in 1:n_iterations
    img = load("phaseplots/phaseplot_$it.png")
    plot(img, size=(720,720), axis=([], false))
end
mp4(phaseplots, fps=1)

absplots = @animate for it in 1:n_iterations
    img = load("abs/abs_$it.png")
    plot(img, size=(720,520), axis=([], false))
end
mp4(absplots, fps=1)

trajplots = @animate for it in 1:n_iterations
    img = load("trajectories/traj_$it.png")
    plot(img, axis=([], false))
end
mp4(trajplots, fps=1)

powerplots = @animate for it in 1:n_iterations
    img = load("power_spectra/power_$it.png")
    plot(img, axis=([], false))
end
mp4(powerplots, fps=1)
