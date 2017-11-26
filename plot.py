# plot.py
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

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, factorial
from myplot import *

## Format the axis in a standard way.
def formatAxis():
    plt.semilogy()
    plt.xlabel("$d$")
    plt.ylabel("$P(d)$")

# Calculate the out name, which is required to save the results of the
# simulation in a meaningfull way.
#
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param step The simulation step that has to be saved.
# @param params Parameter array to generate the graph.
#
# @return The name snippet required to save the results.
def outName(name, numberOfVertices, step, params):
    out = name + "_" + str(numberOfVertices) + "_" + str(step)
    for param in params:
        if (ceil(param) == param):
            out = out + "_" + str(int(ceil(param)))
        else:
            out = out + "_" + str(param)
    
    return(out)

## Plot a single histogram.
#
# @param directory Where should the plot be saved.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs How many graphs have to be generated?
# @param params Parameter array to generate the graph.
# @param fmt Format the plot.
# @param **kwargs Additional named arguments. Make sure all those arguments are valid
#                 according to the matplotlib documentation.
#
# @return The data of the histogram.
def plotHistogram(directory, name, numberOfVertices, numberOfGraphs, params, fmt, **kwargs):
    aux = outName(name, numberOfVertices, numberOfGraphs, params)
    
    # open histogram
    d, pd, err = np.loadtxt(directory + "/hist_" + aux + ".csv", unpack = True, delimiter = " ")
    
    # remove the probabilities that are zero
    d = d[pd != 0]
    err = err[pd != 0]
    pd = pd[pd != 0]
    
    plt.plot(d, pd, fmt, **kwargs)
    
    return([d, pd, err])

## Plot a single histogram with errorbars.
#
# @param directory Where should the plot be saved.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs How many graphs have to be generated?
# @param params Parameter array to generate the graph.
# @param **kwargs Additional named arguments. Make sure all those arguments are valid
#                 according to the matplotlib documentation.
#
# @return The data of the histogram.
def plotHistogramError(directory, name, numberOfVertices, numberOfGraphs, params, **kwargs):
    aux = outName(name, numberOfVertices, numberOfGraphs, params)
    
    # open histogram
    d, pd, err = np.loadtxt(directory + "/hist_" + aux + ".csv", unpack = True, delimiter = " ")
    
    # remove the probabilities that are zero
    d = d[pd != 0]
    err = err[pd != 0]
    pd = pd[pd != 0]
    
    errorbar(d, pd, err, **kwargs)
    
    return([d, pd, err])

## Open the results of the two fits.
#
# @param directory Where should the plot be saved.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs How many graphs have to be generated?
# @param params Parameter array to generate the graph.
def openFitResults(directory, name, numberOfVertices, numberOfGraphs, params):
    aux = outName(name, numberOfVertices, numberOfGraphs, params)
    
    # open fit results pow
    rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW = [0,0,0,0,0,0,0,0,0]
    try:
        rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW = np.loadtxt(directory + "/pow_" + aux + ".csv", unpack = True, delimiter = " ")
    except:
        print("error for " + directory + "/pow_" + aux + ".csv")
    
    # open fit results normal
    rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL = [0,0,0,0,0,0,0]
    try:
        rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL = np.loadtxt(directory + "/normal_" + aux + ".csv", unpack = True, delimiter = " ")
    except:
        print("error for " + directory + "/pow_" + aux + ".csv")
    
    # open fit results poisson
    rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON = [0,0,0,0,0]
    try:
        rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON = np.loadtxt(directory + "/poisson_" + aux + ".csv", unpack = True, delimiter = " ")
    except:
        print("error for " + directory + "/poisson_" + aux + ".csv")
    
    return([rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW, rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL, rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON])

## Plot standard plot (with fits) for the given configuration.
#
# @param directory Where should the plot be saved.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs How many graphs have to be generated?
# @param params Parameter array to generate the graph.
def plot(directory, name, numberOfVertices, numberOfGraphs, params):
    aux = outName(name, numberOfVertices, numberOfGraphs, params)
    
    # open fit results
    [rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW, rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL, rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON] = openFitResults(directory, name, numberOfVertices, numberOfGraphs, params)
    
    # plot the resulting histogram
    plt.close()
    fig,ax = plt.subplots(1)
    
    d, pd, err = plotHistogramError(directory, name, numberOfVertices, numberOfGraphs, params, fmt = "ok", capsize = 5, label = r"result")
    
    sep = len(d) / 1000.0
    dfit = np.arange(1.0, (len(d) + sep), sep)
    dfitInt = np.arange(1, (len(d) + 1), 1)
    fitPow = k * (c / dfit) ** (l * dfit)
    fitPoisson = np.array([lam**x / factorial(x) * np.exp(-lam) for x in dfitInt])
    plt.plot(dfit, fitPow, "r", linestyle = "-.", label = r"$P(d) = k \left(\displaystyle\frac{C}{d}\right)^{\lambda d}$")
    plt.plot(dfitInt, fitPoisson, "g", linestyle = ":", label = r"$P(d) = \displaystyle\frac{\lambda^d}{d!} \cdot \exp\left(-\lambda\right)$")
    
    if (sigma != 0):
        fitNorm = np.exp(-(dfit - mu) ** 2 / (2.0 * sigma ** 2)) / np.sqrt(2.0 * np.pi * sigma ** 2)
        fitNorm = fitNorm[fitNorm != np.Inf]
        dfitN = dfit[fitNorm != np.Inf]
        plt.plot(dfitN, fitNorm, "b", linestyle = "-", label = r"$P(d) = \displaystyle\frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\displaystyle\frac{(d - \mu)^2}{2 \sigma^2} \right)$")
        if (any(fitNorm < 10**(-20))):
            plt.ylim([10**(-20), max(fitNorm) + 1])
    
    # make legend nicer
    handles,labels = ax.get_legend_handles_labels()
    #if (len(handles) == 3):
    #    handles = [handles[2], handles[0], handles[1]]
    #    labels = [labels[2], labels[0], labels[1]]
    #else:
    #    handles = [handles[1], handles[0]]
    #    labels = [labels[1], labels[0]]
    plt.legend(handles, labels)
    formatAxis()
    
    plt.savefig("%s/%s.pdf"%(directory, aux))

plt.rcParams['figure.figsize'] = (7.0, 7.0) # change the size of the plot
plt.rcParams['font.size'] = (14)
plt.rcParams['font.family'] = ('sans')

directory = None

name = "pa"

# Prepare the latex table containing the configuration and chi^2 / df for both fits.
file = open(directory + "/%s_tab.tex"%(name), "w")
for numberOfVertices in [50, 100, 200, 400]:
    for numberOfGraphs in [1000, 3500, 6000, 8500, 10000]:
        for numEdges in [1, 2]:
            # open fit results
            rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW, rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL, rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON = openFitResults(directory, name, numberOfVertices, numberOfGraphs, [numEdges])
            file.write(str(numberOfVertices) + ' & ' + str(numberOfGraphs) + ' & ' + str(numEdges) + ' & \\num{' + '{:0.3e}'.format(rPOW) + '} & \\num{' + '{:0.3e}'.format(rNORMAL) + '} & \\num{' + '{:0.3e}'.format(rPOISSON) + '} \\\\\n')
            plot(directory, name, numberOfVertices, numberOfGraphs, [numEdges])
file.close()

# plot numberOfGraphs = 10000
plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m], "+b", label = r"$\left|V\right| = 50,~m = 1$")
plotHistogram(directory, name, 100, 10000, [m], "+k", label = r"$\left|V\right| = 100,~m = 1$")
plotHistogram(directory, name, 200, 10000, [m], "+r", label = r"$\left|V\right| = 200,~m = 1$")
plotHistogram(directory, name, 400, 10000, [m], "+m", label = r"$\left|V\right| = 400,~m = 1$")

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m], "*b", label = r"$\left|V\right| = 50,~m = 2$")
plotHistogram(directory, name, 100, 10000, [m], "*k", label = r"$\left|V\right| = 100,~m = 2$")
plotHistogram(directory, name, 200, 10000, [m], "*r", label = r"$\left|V\right| = 200,~m = 2$")
plotHistogram(directory, name, 400, 10000, [m], "*m", label = r"$\left|V\right| = 400,~m = 2$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s.pdf"%(directory, name, str(numberOfGraphs)))

numberOfVertices = 50
m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

numberOfVertices = 100
m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

numberOfVertices = 200
m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

numberOfVertices = 400
m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s.pdf"%(directory, name, str(numberOfVertices), str(m)))






name = "ca"

# Prepare the latex table containing the configuration and chi^2 / df for both fits.
file = open(directory + "/%s_tab.tex"%(name), "w")
for numberOfVertices in [50, 100, 200, 400]:
    for numberOfGraphs in [1000, 3500, 6000, 8500, 10000]:
        for numEdges in [1, 2]:
            for probability in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                # open fit results
                rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW, rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL, rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON = openFitResults(directory, name, numberOfVertices, numberOfGraphs, [numEdges, probability])
                file.write(str(numberOfVertices) + ' & ' + str(numberOfGraphs) + ' & ' + str(numEdges) + ' & ' + str(probability) + ' & \\num{' + '{:0.3e}'.format(rPOW) + '} & \\num{' + '{:0.3e}'.format(rNORMAL) + '} & \\num{' + '{:0.3e}'.format(rPOISSON) + '} \\\\\n')
                plot(directory, name, numberOfVertices, numberOfGraphs, [numEdges, probability])
file.close()

numberOfGraphs = 10000

probability = 0.1

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.2

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.3

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.4

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.5

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.6

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.7

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.8

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

probability = 0.9

plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m, probability], "+b", label = r"$\left|V\right| = 50,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "+k", label = r"$\left|V\right| = 100,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "+r", label = r"$\left|V\right| = 200,~m = 1,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "+m", label = r"$\left|V\right| = 400,~m = 1,~\alpha = %s$"%(str(probability)))

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m, probability], "*b", label = r"$\left|V\right| = 50,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 100, 10000, [m, probability], "*k", label = r"$\left|V\right| = 100,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 200, 10000, [m, probability], "*r", label = r"$\left|V\right| = 200,~m = 2,~\alpha = %s$"%(str(probability)))
plotHistogram(directory, name, 400, 10000, [m, probability], "*m", label = r"$\left|V\right| = 400,~m = 2,~\alpha = %s$"%(str(probability)))
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s_const-probability_%s.pdf"%(directory, name, str(numberOfGraphs), str(probability)))

numberOfVertices = 50

probability = 0.1

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.2

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.3

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.4

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.5

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.6

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.7

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.8

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.9

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

numberOfVertices = 100

probability = 0.1

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.2

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.3

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.4

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.5

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.6

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.7

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.8

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.9

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

numberOfVertices = 200

probability = 0.1

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.2

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.3

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.4

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.5

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.6

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.7

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.8

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.9

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

numberOfVertices = 400

probability = 0.1

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.2

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.3

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.4

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.5

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.6

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.7

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.8

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

probability = 0.9

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, name, numberOfVertices, 1000, [m, probability], "+b", label = r"$N = 1000$")
plotHistogram(directory, name, numberOfVertices, 3500, [m, probability], "ok", label = r"$N = 3500$")
plotHistogram(directory, name, numberOfVertices, 6000, [m, probability], "*r", label = r"$N = 6000$")
plotHistogram(directory, name, numberOfVertices, 8500, [m, probability], "vm", label = r"$N = 8500$")
plotHistogram(directory, name, numberOfVertices, 10000, [m, probability], "xy", label = r"$N = 10000$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfVertices_%s-m_%s-p_%s.pdf"%(directory, name, str(numberOfVertices), str(m), str(probability)))





name = "er"





# Prepare the latex table containing the configuration and chi^2 / df for both fits.
file = open(directory + "/%s_tab.tex"%(name), "w")
for numberOfVertices in [50, 100, 200, 400]:
    for numberOfGraphs in [1000, 3500, 6000, 8500, 10000]:
        for numEdges in [1, 2]:
            # open fit results
            rPOW, chisquaredfPOW, k, kerr, c, cerr, l, lerr, dfPOW, rNORMAL, chisquaredfNORMAL, mu, muerr, sigma, sigmaerr, dfNORMAL, rPOISSON, chisquaredfPOISSON, lam, lamerr, dfPOISSON = openFitResults(directory, name, numberOfVertices, numberOfGraphs, [numEdges])
            file.write(str(numberOfVertices) + ' & ' + str(numberOfGraphs) + ' & ' + str(numEdges) + ' & \\num{' + '{:0.3e}'.format(rPOW) + '} & \\num{' + '{:0.3e}'.format(rNORMAL) + '} & \\num{' + '{:0.3e}'.format(rPOISSON) + '} \\\\\n')
            plot(directory, name, numberOfVertices, numberOfGraphs, [numEdges])
file.close()

# plot numberOfGraphs = 10000
plt.close()
fig,ax = plt.subplots(1)
# plot m = 1
m = 1
plotHistogram(directory, name, 50, 10000, [m], "+b", label = r"$\left|V\right| = 50,~m = 1$")
plotHistogram(directory, name, 100, 10000, [m], "+k", label = r"$\left|V\right| = 100,~m = 1$")
plotHistogram(directory, name, 200, 10000, [m], "+r", label = r"$\left|V\right| = 200,~m = 1$")
plotHistogram(directory, name, 400, 10000, [m], "+m", label = r"$\left|V\right| = 400,~m = 1$")

# plot m = 2
m = 2
plotHistogram(directory, name, 50, 10000, [m], "*b", label = r"$\left|V\right| = 50,~m = 2$")
plotHistogram(directory, name, 100, 10000, [m], "*k", label = r"$\left|V\right| = 100,~m = 2$")
plotHistogram(directory, name, 200, 10000, [m], "*r", label = r"$\left|V\right| = 200,~m = 2$")
plotHistogram(directory, name, 400, 10000, [m], "*m", label = r"$\left|V\right| = 400,~m = 2$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_const-numberOfGraphs_%s.pdf"%(directory, name, str(numberOfGraphs)))






numberOfVertices = 50

plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.1], "+b", label = r"$m = 1,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.2], "+k", label = r"$m = 1,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.3], "+r", label = r"$m = 1,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.4], "+m", label = r"$m = 1,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.5], "+y", label = r"$m = 1,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.6], "+g", label = r"$m = 1,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.7], "+c", label = r"$m = 1,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.8], "+", label = r"$m = 1,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.9], "+", label = r"$m = 1,~\alpha = 0.9$", color = "silver")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.1], "*b", label = r"$m = 2,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.2], "*k", label = r"$m = 2,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.3], "*r", label = r"$m = 2,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.4], "*m", label = r"$m = 2,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.5], "*y", label = r"$m = 2,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.6], "*g", label = r"$m = 2,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.7], "*c", label = r"$m = 2,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.8], "*", label = r"$m = 2,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.9], "*", label = r"$m = 2,~\alpha = 0.9$", color = "silver")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_all_%s_10000.pdf"%(directory, name, str(numberOfVertices)))

numberOfVertices = 100

plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.1], "+b", label = r"$m = 1,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.2], "+k", label = r"$m = 1,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.3], "+r", label = r"$m = 1,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.4], "+m", label = r"$m = 1,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.5], "+y", label = r"$m = 1,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.6], "+g", label = r"$m = 1,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.7], "+c", label = r"$m = 1,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.8], "+", label = r"$m = 1,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.9], "+", label = r"$m = 1,~\alpha = 0.9$", color = "silver")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.1], "*b", label = r"$m = 2,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.2], "*k", label = r"$m = 2,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.3], "*r", label = r"$m = 2,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.4], "*m", label = r"$m = 2,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.5], "*y", label = r"$m = 2,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.6], "*g", label = r"$m = 2,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.7], "*c", label = r"$m = 2,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.8], "*", label = r"$m = 2,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.9], "*", label = r"$m = 2,~\alpha = 0.9$", color = "silver")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_all_%s_10000.pdf"%(directory, name, str(numberOfVertices)))

numberOfVertices = 200

plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.1], "+b", label = r"$m = 1,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.2], "+k", label = r"$m = 1,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.3], "+r", label = r"$m = 1,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.4], "+m", label = r"$m = 1,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.5], "+y", label = r"$m = 1,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.6], "+g", label = r"$m = 1,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.7], "+c", label = r"$m = 1,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.8], "+", label = r"$m = 1,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.9], "+", label = r"$m = 1,~\alpha = 0.9$", color = "silver")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.1], "*b", label = r"$m = 2,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.2], "*k", label = r"$m = 2,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.3], "*r", label = r"$m = 2,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.4], "*m", label = r"$m = 2,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.5], "*y", label = r"$m = 2,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.6], "*g", label = r"$m = 2,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.7], "*c", label = r"$m = 2,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.8], "*", label = r"$m = 2,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.9], "*", label = r"$m = 2,~\alpha = 0.9$", color = "silver")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_all_%s_10000.pdf"%(directory, name, str(numberOfVertices)))

numberOfVertices = 400

plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.1], "+b", label = r"$m = 1,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.2], "+k", label = r"$m = 1,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.3], "+r", label = r"$m = 1,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.4], "+m", label = r"$m = 1,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.5], "+y", label = r"$m = 1,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.6], "+g", label = r"$m = 1,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.7], "+c", label = r"$m = 1,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.8], "+", label = r"$m = 1,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [1, 0.9], "+", label = r"$m = 1,~\alpha = 0.9$", color = "silver")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.1], "*b", label = r"$m = 2,~\alpha = 0.1$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.2], "*k", label = r"$m = 2,~\alpha = 0.2$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.3], "*r", label = r"$m = 2,~\alpha = 0.3$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.4], "*m", label = r"$m = 2,~\alpha = 0.4$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.5], "*y", label = r"$m = 2,~\alpha = 0.5$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.6], "*g", label = r"$m = 2,~\alpha = 0.6$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.7], "*c", label = r"$m = 2,~\alpha = 0.7$")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.8], "*", label = r"$m = 2,~\alpha = 0.8$", color = "orange")
plotHistogram(directory, "ca", numberOfVertices, 10000, [2, 0.9], "*", label = r"$m = 2,~\alpha = 0.9$", color = "silver")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/%s_all_%s_10000.pdf"%(directory, name, str(numberOfVertices)))

m = 1
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "pa", 400, 10000, [m], "+b", label = r"pa")
plotHistogram(directory, "ca", 400, 10000, [m, 0.1], "ok", label = r"$\alpha = 0.1$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.2], "*r", label = r"$\alpha = 0.2$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.3], "vm", label = r"$\alpha = 0.3$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.4], "xy", label = r"$\alpha = 0.4$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.5], "^g", label = r"$\alpha = 0.5$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.6], "vc", label = r"$\alpha = 0.6$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.7], "<", label = r"$\alpha = 0.7$", color = "orange")
plotHistogram(directory, "ca", 400, 10000, [m, 0.8], ">", label = r"$\alpha = 0.8$", color = "silver")
plotHistogram(directory, "ca", 400, 10000, [m, 0.9], "Db", label = r"$\alpha = 0.9$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/comp_ca-pa_m_%s.pdf"%(directory, str(m)))

m = 2
plt.close()
fig,ax = plt.subplots(1)
plotHistogram(directory, "pa", 400, 10000, [m], "+b", label = r"pa")
plotHistogram(directory, "ca", 400, 10000, [m, 0.1], "ok", label = r"$\alpha = 0.1$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.2], "*r", label = r"$\alpha = 0.2$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.3], "vm", label = r"$\alpha = 0.3$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.4], "xy", label = r"$\alpha = 0.4$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.5], "^g", label = r"$\alpha = 0.5$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.6], "vc", label = r"$\alpha = 0.6$")
plotHistogram(directory, "ca", 400, 10000, [m, 0.7], "<", label = r"$\alpha = 0.7$", color = "orange")
plotHistogram(directory, "ca", 400, 10000, [m, 0.8], ">", label = r"$\alpha = 0.8$", color = "silver")
plotHistogram(directory, "ca", 400, 10000, [m, 0.9], "Db", label = r"$\alpha = 0.9$")
formatAxis()
plt.semilogy()
plt.legend()
plt.savefig("%s/comp_ca-pa_m_%s.pdf"%(directory, str(m)))
