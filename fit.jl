# fit.jl
# Copyright (C) flossCoder
#
# This file is part of shortestPath.
#
# shortestPath is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# shortestPath is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

include("outName.jl")

module Fit

using OutName

export FitType, pow, normal, poisson, doFit, saveFit, mainFit

using LsqFit # fit library: https://github.com/JuliaNLSolvers/LsqFit.jl

# Collect some statistical results of the fit.
#
# @param fit An object of the LsqFitResult type. Paramesers according documentation
#            (https://github.com/JuliaNLSolvers/LsqFit.jl)
#            fit.dof: degrees of freedom
#            fit.param: best fit parameters
#            fit.resid: residuals = vector of residuals
#            fit.jacobian: estimated Jacobian at solution
# @param errors Estimated errors of the fit parameters.
# @param chisquaredf The chi^2 / df value of the fit ("goodness of fit").
mutable struct FitType
    fit # see LsqFitResult type
    errors
    chisquaredf::Float64
    r::Float64 # correlation coefficient
end

# Calculate the power function P(d) = k * (C / d)^(lambda * d)
#
# @param d An array of the independent variable.
# @param parameters An array containing the fit parameter:
#        parameters[1] = k
#        parameters[2] = C
#        parameters[3] = lambda
#
# @return The resulting array.
pow(d, parameters) = parameters[1] .* (parameters[2] ./ d).^(parameters[3] .* d)

# Calculate the normaldistripution P(d) = 1 / sqrt(2 * pi * sigma^2) * exp(-(d - mu)^2 / (2 * sigma^2))
#
# @param d An array of the independent variable.
# @param parameters An array containing the fit parameter:
#        parameters[1] = mu
#        parameters[2] = sigma
#
# @return The resulting array.
normal(d, parameters) = exp.(-(d - parameters[1]).^2 ./ (2.0 .* parameters[2] .^ 2)) ./ sqrt.(2.0 .* pi .* parameters[2].^2)

# Calculate the poisson distribution P(d) = lambda^d / d! * exp(-lambda)
#
# @param d An array of the independent variable.
# @param parameters An array containing the fit parameter:
#        parameters[1] = lambda
#
# @return The resulting array.
poisson(d, parameters) = parameters[1].^d ./ factorial.(d) .* exp.(-parameters[1])

# Do the fit and calculate the chi^2 / df value.
#
# @param model Function handle with model(x, parameter-array).
# @param xdata The x data-array for the fit.
# @param ydata The y data-array for the fit.
# @param py An array containing the starting condition of the fit-parameters.
# @param ceb Change the probability of the 95 % confidence error bars of the
#            fit parameter.
#
# @return The fitting results as FitType-object.
function doFit(model, xdata, ydata, p0, ceb = 0.95)
    fit = curve_fit(model, xdata, ydata, p0)
    errors = estimate_errors(fit, ceb)
    chisquaredf = sum((fit.resid).^2) / fit.dof
    fitData = model(xdata, fit.param)
    r = cov(ydata, fitData) / (sqrt(var(ydata) * var(fitData)))
    return(FitType(fit, errors, chisquaredf, r))
end

# Save the fit results.
#
# @param outName Provide a name for the output file.
# @param result The fitting results as FitType-object.
function saveFit(outName::String, result::FitType)
    output = "$(result.r) $(result.chisquaredf)"
    for index = 1:length(result.errors)
        output = "$output $(result.fit.param[index]) $(result.errors[index])"
    end
    output = "$output $(result.fit.dof)"
    outFile = open(outName, "w")
    write(outFile, output)
    close(outFile)
end

# Open the given configurations, perform the fits and save the results.
#
# @param directory Valid directory to save histogram PDF's, plots and fit results.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs An array containing the number of graphs for saving results.
# @param params Parameter array to generate the graph.
# @param isDigraph true, if the graph is a digraph, false otherwise (default = false).
function mainFit(directory::String, name::String, numberOfVertices::Int, numberOfGraphs::Int, params, isDigraph = false)
    # Open the pdf of the histogram.
    theName = outName(name, numberOfVertices, numberOfGraphs, params)
    pdf = readdlm(string(directory, "/", "hist_", theName, ".csv"))

    d = pdf[1:size(pdf)[1], 1]
    ydata = pdf[1:size(pdf)[1], 2]

    # remove all ydata == 0
    index = findfirst(ydata, 0)
    while (index != 0)
        deleteat!(d, index)
        deleteat!(ydata, index)
        index = findfirst(ydata, 0)
    end

    maxP = maximum(ydata)

    # determine the x value of maxP
    maxPindex = 1
    while (pdf[maxPindex, 2] != maxP)
        maxPindex += 1
    end

    try
        powResult = doFit(pow, d, ydata, [maxP, 1.0, 1.0])
        saveFit(string(directory, "/", "pow_", theName, ".csv"), powResult)
    catch
        outFile = open(string(directory, "/", "pow_", theName, ".csv"), "w")
        write(outFile, "0 0 0 0 0 0 0 0 0")
        close(outFile)
        println(string(directory, "/", "pow_", theName, ".csv"))
    end

    try
        normalResult = doFit(normal, d, ydata, [maxPindex, maxP])
        saveFit(string(directory, "/", "normal_", theName, ".csv"), normalResult)
    catch
        outFile = open(string(directory, "/", "normal_", theName, ".csv"), "w")
        write(outFile, "0 0 0 0 0 0 0")
        close(outFile)
        println(string(directory, "/", "normal_", theName, ".csv"))
    end

    try
        poissonResult = doFit(poisson, d, ydata, [1.0])
        saveFit(string(directory, "/", "poisson_", theName, ".csv"), poissonResult)
    catch
        outFile = open(string(directory, "/", "poisson_", theName, ".csv"), "w")
        write(outFile, "0 0 0 0 0")
        close(outFile)
        println(string(directory, "/", "poisson_", theName, ".csv"))
    end
end

end
