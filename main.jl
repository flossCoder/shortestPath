# main.jl
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

include("graph.jl")
include("simulation.jl")
include("fit.jl")

using Graph
using Simulation
using Fit

# set seed for the Mersenne Twister random number generator
srand(42)

numberOfVertices = [50, 100, 200, 400]
numberOfGraphs = [1000, 3500, 6000, 8500, 10000]
probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
directory = ""

# run the simulation

for size in numberOfVertices
    for numEdges = 1:2
        doSimulation(directory, "pa", size, numberOfGraphs, preferentialAttachment, [numEdges])
        doSimulation(directory, "er", size, numberOfGraphs, generateER, [numEdges])
        for probability in probabilities
            doSimulation(directory, "ca", size, numberOfGraphs, copyAttachment, [numEdges, probability])
        end
    end
end

# do the fitting business

for size in numberOfVertices
    for sweeps in numberOfGraphs
        for numEdges = 1:2
            mainFit(directory, "pa", size, sweeps, [numEdges])
            mainFit(directory, "er", size, sweeps, [numEdges])
            for probability in probabilities
                mainFit(directory, "ca", size, sweeps, [numEdges, probability])
            end
        end
    end
end

println("Simulations done")
