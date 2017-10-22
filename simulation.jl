# simulation.jl
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
include("histogram.jl")
include("outName.jl")

module Simulation

using Graph
using Histogram
using OutName

export doSimulation

# Insert the shortest paths into the histogram.
#
# @param numberOfVertices Contains the number of verticises of the graph.
# @param shortest a n x n Matrix containing the pairwise shortest paths
# @param histogram The histogram containing the data.
#
# @return The histogram after inserting the given shortest matrix
function shortestInHistogram(numberOfVertices::Int, shortest, histogram::HistogramType)
    for index1 = (1:numberOfVertices)
        for index2 = (1:numberOfVertices)
            if (shortest[index1, index2] != Inf)
                histogram = insertHistogram(histogram, shortest[index1, index2])
            end
        end
    end
    return(histogram)
end

# Do the simulation.
#
# @param directory Valid directory to save histogram PDF's, plots and fit results.
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param numberOfGraphs An array containing the number of graphs for saving results.
# @param generator A function handle that generates the new graphs during the simulation.
#                  generator is allways called in the following way:
#                  generator(graph, params)
# @param params Parameter array to generate the graph.
# @param isDigraph true, if the graph is a digraph, false otherwise (default = false).
function doSimulation(directory::String, name::String, numberOfVertices::Int, numberOfGraphs, generator, params, isDigraph = false)
    # Initialize the histogram. The longest feasible shortest path contains numberOfVertices - 1
    # many vertices (under the assumption, that all edge weights are one).
    histogram = initHistogram(0.5, Float64(numberOfVertices), 1.0)
    step = 1
    current = 1 # index of the current numberOfGraphs array entry
    while (current <= length(numberOfGraphs))
        # generate a new (empty) graph
        graph = initGraph(isDigraph, numberOfVertices)
        # do the preferential attachment
        graph = generator(graph, params)
        # calculate all shortest pairs via the Floyd-Warshall algorithm
        shortest = allPairShortestPath(graph)
        # fill the histogram
        histogram = shortestInHistogram(numberOfVertices, shortest, histogram)

        if (step == numberOfGraphs[current])
            # evalulation and saving results has to be done
            # calculate the pdf (plus standard error) of the histogram
            histogramPDF = calculatePDFHistogram(histogram)

            # Save the pdf as csv-file.
            theName = outName(name, numberOfVertices, step, params)
            outFile = open(string(directory, "/", "hist_", theName, ".csv"), "w")
            for index = 1:histogramPDF.numberOfBins
                write(outFile, "$(index) $(histogramPDF.pdf[index]) $(histogramPDF.stderr[index])\n")
            end
            close(outFile)

            current += 1
        end
        step += 1
    end
end

end
