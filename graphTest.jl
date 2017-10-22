# graphTest.jl
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

module GraphTest

using Graph

# Test initGraph
graph1 = initGraph(false, 10)

if (graph1.isDigraph == true)
    println("Test initGraph failed")
end

if (graph1.numberOfVertices != 10)
    println("Test initGraph failed")
end

if (graph1.numberOfEdges != 0)
    println("Test initGraph failed")
end

if (graph1.adjacencyMatrix != zeros(10, 10) + Inf)
    println("Test initGraph failed")
end

graph2 = initGraph(true, 10)

if (graph2.isDigraph == false)
    println("Test initGraph failed")
end

if (graph2.numberOfVertices != 10)
    println("Test initGraph failed")
end

if (graph2.numberOfEdges != 0)
    println("Test initGraph failed")
end

if (graph2.adjacencyMatrix != zeros(10, 10) + Inf)
    println("Test initGraph failed")
end

# Test insertEdge
graph1 = insertEdge(graph1, 4, 6)

aux1 = zeros(10, 10) + Inf
aux1[4, 6] = 1

if (graph1.numberOfEdges != 1)
    println("Test insertEdge failed")
end

if (graph1.adjacencyMatrix != aux1)
    println("Test insertEdge failed")
end

graph2 = insertEdge(graph2, 3, 5)

aux2 = zeros(10, 10) + Inf
aux2[3, 5] = 1
aux2[5, 3] = 1

if (graph2.numberOfEdges != 1)
    println("Test insertEdge failed")
end

if (graph2.adjacencyMatrix != aux2)
    println("Test insertEdge failed")
end

# Test edgeExists.
for i = 1:10
    for j = 1:10
        if (edgeExists(graph1, i, j) && (aux1[i, j] == Inf))
            println("Test edgeExists failed")
        end
        if (!edgeExists(graph1, i, j) && (aux1[i, j] == 1))
            println("Test edgeExists failed")
        end

        if (edgeExists(graph2, i, j) && (aux2[i, j] == Inf))
            println("Test edgeExists failed")
        end
        if (!edgeExists(graph2, i, j) && (aux2[i, j] == 1))
            println("Test edgeExists failed")
        end
    end
end


# Test remove edge
graph1 = removeEdge(graph1, 8, 2)

if (graph1.numberOfEdges != 1)
    println("Test removeEdge failed")
end

if (graph1.adjacencyMatrix != aux1)
    println("Test removeEdge failed")
end

graph1 = removeEdge(graph1, 4, 6)

if (graph1.numberOfEdges != 0)
    println("Test removeEdge failed")
end

if (graph1.adjacencyMatrix != zeros(10, 10) + Inf)
    println("Test removeEdge failed")
end

graph2 = removeEdge(graph2, 7, 3)

if (graph2.numberOfEdges != 1)
    println("Test removeEdge failed")
end

if (graph2.adjacencyMatrix != aux2)
    println("Test removeEdge failed")
end

graph2 = removeEdge(graph2, 3, 5)

if (graph2.numberOfEdges != 0)
    println("Test removeEdge failed")
end

if (graph2.adjacencyMatrix != zeros(10, 10) + Inf)
    println("Test removeEdge failed")
end

# Test preferential attachment.
# set seed for the Mersenne Twister random number generator
srand(42)

# set up auxiliary graph
aux1 = initGraph(false, 10)
aux1 = insertEdge(aux1, 1, 2)
aux1 = insertEdge(aux1, 1, 3)
aux1 = insertEdge(aux1, 1, 4)
aux1 = insertEdge(aux1, 2, 3)
aux1 = insertEdge(aux1, 2, 4)
aux1 = insertEdge(aux1, 3, 4)
aux1 = insertEdge(aux1, 5, 2)
aux1 = insertEdge(aux1, 5, 1)
aux1 = insertEdge(aux1, 5, 4)
aux1 = insertEdge(aux1, 6, 5)
aux1 = insertEdge(aux1, 6, 3)
aux1 = insertEdge(aux1, 6, 4)
aux1 = insertEdge(aux1, 7, 5)
aux1 = insertEdge(aux1, 7, 4)
aux1 = insertEdge(aux1, 7, 3)
aux1 = insertEdge(aux1, 8, 4)
aux1 = insertEdge(aux1, 8, 2)
aux1 = insertEdge(aux1, 8, 6)
aux1 = insertEdge(aux1, 9, 3)
aux1 = insertEdge(aux1, 9, 2)
aux1 = insertEdge(aux1, 9, 5)
aux1 = insertEdge(aux1, 10, 9)
aux1 = insertEdge(aux1, 10, 3)
aux1 = insertEdge(aux1, 10, 4)

aux2 = initGraph(true, 10)
aux2 = insertEdge(aux2, 1, 2)
aux2 = insertEdge(aux2, 1, 3)
aux2 = insertEdge(aux2, 1, 4)
aux2 = insertEdge(aux2, 1, 5)
aux2 = insertEdge(aux2, 2, 3)
aux2 = insertEdge(aux2, 2, 4)
aux2 = insertEdge(aux2, 2, 5)
aux2 = insertEdge(aux2, 3, 4)
aux2 = insertEdge(aux2, 3, 5)
aux2 = insertEdge(aux2, 4, 5)
aux2 = insertEdge(aux2, 6, 3)
aux2 = insertEdge(aux2, 6, 1)
aux2 = insertEdge(aux2, 6, 5)
aux2 = insertEdge(aux2, 6, 4)
aux2 = insertEdge(aux2, 7, 4)
aux2 = insertEdge(aux2, 7, 1)
aux2 = insertEdge(aux2, 7, 6)
aux2 = insertEdge(aux2, 7, 2)
aux2 = insertEdge(aux2, 8, 4)
aux2 = insertEdge(aux2, 8, 7)
aux2 = insertEdge(aux2, 8, 2)
aux2 = insertEdge(aux2, 8, 1)
aux2 = insertEdge(aux2, 9, 1)
aux2 = insertEdge(aux2, 9, 2)
aux2 = insertEdge(aux2, 9, 4)
aux2 = insertEdge(aux2, 9, 6)
aux2 = insertEdge(aux2, 10, 4)
aux2 = insertEdge(aux2, 10, 2)
aux2 = insertEdge(aux2, 10, 9)
aux2 = insertEdge(aux2, 10, 5)

graph1 = preferentialAttachment(graph1, [3])

if (graph1.adjacencyMatrix != aux1.adjacencyMatrix)
    println("Test preferentialAttachment failed")
end

graph2 = preferentialAttachment(graph2, [4])

if (graph2.adjacencyMatrix != aux2.adjacencyMatrix)
    println("Test preferentialAttachment failed")
end

# Test of Floyd-Warshall algorithm with an example from:
# https://www.informatik.hu-berlin.de/de/forschung/gebiete/algorithmenII/Lehre/ss09/theo3/skript/thi3-bsp-fw1.pdf

adj = [Inf 2 Inf Inf Inf; Inf Inf Inf Inf -3; Inf -2 Inf Inf Inf; Inf Inf 4 Inf Inf; 10 Inf 9 1 Inf]
dResult = [9 2 4 0.0 -1; 7 0.0 2 -2 -3; 5 -2 0.0 -4 -5; 9 2 4 0.0 -1; 10 3 5 1 0.0]

testGraph = initGraph(false, 5)
testGraph.adjacencyMatrix = adj # trick: change the adjacency matrix such that the example fits
testGraph.numberOfEdges = 7

shortest = allPairShortestPath(testGraph)

if (shortest != dResult)
    println("Test allPairShortestPath failed")
end

# Test the generateEdgeList function.
if (generateEdgeList(testGraph, 1) != [2.0])
    println("Test generateEdgeList failed")
end

if (generateEdgeList(testGraph, 2) != [5])
    println("Test generateEdgeList failed")
end

if (generateEdgeList(testGraph, 3) != [2])
    println("Test generateEdgeList failed")
end

if (generateEdgeList(testGraph, 4) != [3])
    println("Test generateEdgeList failed")
end

if (generateEdgeList(testGraph, 5) != [1 3 4])
    println("Test generateEdgeList failed")
end

println("GraphTest done")
end
