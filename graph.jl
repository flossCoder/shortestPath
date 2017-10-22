# graph.jl
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

module Graph

export GraphType, initGraph, insertEdge, edgeExists, removeEdge, preferentialAttachment, allPairShortestPath, generateEdgeList, copyAttachment

# Define a graph.
#
# @param isDigraph true, if the graph is a digraph, false otherwise.
# @param numberOfVertices Contains the number of vertices of the graph.
# @param numberOfEdges Contains the number of edges of the graph.
mutable struct GraphType
    isDigraph::Bool
    numberOfVertices::Int
    numberOfEdges::Int
    adjacencyMatrix
end

# Initialize a new graph.
#
# @param isDigraph true, if the graph is a digraph, false otherwise.
# @param numberOfVertices Contains the number of verticises of the graph.
#
# @return A new graph object.
function initGraph(isDigraph::Bool, numberOfVertices::Int)
    return(GraphType(isDigraph, numberOfVertices, 0, zeros(numberOfVertices, numberOfVertices) + Inf))
end

# Insert a new edge.
#
# @param graph The graph where one wants to insert an edge.
# @param vertex1 The first vertex.
# @param vertex2 The second vertex.
# @param weight Opional weight (default = 1).
#
# @return The graph after inserting an edge.
function insertEdge(graph::GraphType, vertex1::Int, vertex2::Int, weight = 1)
    if (graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] == Inf && graph.adjacencyMatrix[vertex2, vertex1] == Inf)
        graph.adjacencyMatrix[vertex1, vertex2] = weight
        graph.adjacencyMatrix[vertex2, vertex1] = weight
        graph.numberOfEdges = graph.numberOfEdges + 1
    elseif (!graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] == Inf)
        graph.adjacencyMatrix[vertex1, vertex2] = weight
        graph.numberOfEdges = graph.numberOfEdges + 1
    end

    return(graph)
end

# Does the given edge exist.
#
# @param graph The graph where one wants to check an edge.
# @param vertex1 The first vertex.
# @param vertex2 The second vertex.
#
# @return True: Edge exists, false: otherwise
function edgeExists(graph::GraphType, vertex1::Int, vertex2::Int)
    if (graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] != Inf && graph.adjacencyMatrix[vertex2, vertex1] != Inf)
        return(true) # edge exists in digraph
    elseif (!graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] != Inf)
        return(true) # edge exists in non-digraph
    else
        return(false)
    end
end

# Remove an edge
#
# @param graph The graph where one wants to insert an edge.
# @param vertex1 The first vertex.
# @param vertex2 The second vertex.
#
# @return The graph after removing an edge.
function removeEdge(graph::GraphType, vertex1::Int, vertex2::Int)
    if (graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] != Inf && graph.adjacencyMatrix[vertex2, vertex1] != Inf)
        graph.adjacencyMatrix[vertex1, vertex2] = Inf
        graph.adjacencyMatrix[vertex2, vertex1] = Inf
        graph.numberOfEdges = graph.numberOfEdges - 1
    elseif (!graph.isDigraph && graph.adjacencyMatrix[vertex1, vertex2] != Inf)
        graph.adjacencyMatrix[vertex1, vertex2] = Inf
        graph.numberOfEdges = graph.numberOfEdges - 1
    end

    return(graph)
end

# Implement preferential attachment to insert a number of edges, such that one
# generates a scale-free graph.
#
# @param graph The graph where one wants to insert edges.
# @param params Parameter array to generate the graph. For preferential attachment:
#               The number of edges that should be inserted.
#
# @return A scale-free graph.
function preferentialAttachment(graph::GraphType, params)
    numEdges = params[1]
    # Check, whether a valid numEdges is given.
    if (graph.numberOfVertices < (numEdges + 1))
        error("Invalid numEdges ", string(numEdges), " > numberOfVertices ", string(graph.numberOfVertices))
    end

    # set up an auxiliary array, that marks the inserted edges
    insertedEdges = zeros(2 * numEdges * graph.numberOfVertices - numEdges * (numEdges + 1))
    numInserted = 0
    # initialize a complete subgraph of numEdges + 1 vertices
    for vertex1 = 1:(numEdges + 1)
        for vertex2 = (vertex1 + 1):(numEdges + 1)
            # insert the edge
            graph = insertEdge(graph, vertex1, vertex2)
            # mark the edge
            insertedEdges[(numInserted + 1)] = vertex1
            insertedEdges[(numInserted + 2)] = vertex2
            numInserted += 2
        end
    end

    # add each vertex, that has not been added during the previous initialization
    # according to preferential attachment
    for vertex1 = (numEdges + 2):graph.numberOfVertices
        # add numEdges for each vertex
        counter = 0
        while (counter < numEdges)
            # choose second vertex randomly, such that vertex1 != vertex2
            vertex2 = vertex1
            while (vertex2 == vertex1)
                vertex2 = Int(insertedEdges[Int(floor(rand() * numInserted) + 1)])
            end
            # does the edge in question exist
            if (!edgeExists(graph, vertex1, vertex2))
                # edge doesn't exist => insert it
                graph = insertEdge(graph, vertex1, vertex2)
                # mark the edge
                insertedEdges[(numInserted + 1)] = vertex1
                insertedEdges[(numInserted + 2)] = vertex2
                numInserted += 2
                counter += 1
            end
        end
    end

    return(graph)
end

# Generate a list containing all vertices involved in an edge with the given vertex.
#
# @param graph The graph where one wants to insert edges.
# @param vertex The index of the given vertex.
#
# @return An array containing all neighbours of the given vertex.
function generateEdgeList(graph::GraphType, vertex::Int)
    listOfEdges = []
    for i in 1:graph.numberOfVertices
        if (edgeExists(graph, vertex, i))
            if (listOfEdges == [])
                listOfEdges = [i]
            else
                listOfEdges = [listOfEdges i]
            end
        end
    end
    return(listOfEdges)
end

# Implement a copy attachment according to:
# Kumar, Ravi; Raghavan, Prabhakar (2000). Stochastic Models for the Web Graph.
# Foundations of Computer Science, 41st Annual Symposium on. pp. 57–65.
# doi:10.1109/SFCS.2000.892065.
#
# @param graph The graph where one wants to insert edges.
# @param params Parameter array to generate the graph. For copy attachment:
#               The number of edges that should be inserted.
#               Copy factor.
#
# @return A scale-free graph.
function copyAttachment(graph::GraphType, params)
    numEdges = Int(params[1])
    alpha = params[2]

    # initialize a complete subgraph of numEdges + 1 vertices
    for vertex1 = 1:(numEdges + 1)
        for vertex2 = (vertex1 + 1):(numEdges + 1)
            # insert the edge
            graph = insertEdge(graph, Int(vertex1), Int(vertex2))
        end
    end

    # insert the rest of the vertices
    for vertex1 = (numEdges + 2):graph.numberOfVertices
        prototype = Int(floor(rand() * (vertex1 - 1)) + 1)
        prototypeEdgesList = generateEdgeList(graph, prototype)
        for i = 1:numEdges
            if (rand() < alpha)
                # choose edge randomly
                vertex2 = Int(floor(rand() * (vertex1 - 1)) + 1)
                while (edgeExists(graph, vertex1, vertex2))
                    vertex2 = Int(floor(rand() * (vertex1 - 1)) + 1)
                end
                graph = insertEdge(graph, vertex1, vertex2)
            else
                # copy edge of the prototype
                if (length(prototypeEdgesList) < i)
                    i = i - 1
                elseif (edgeExists(graph, vertex1, prototypeEdgesList[i]))
                    i = i - 1
                else
                    insertEdge(graph, vertex1, prototypeEdgesList[i])
                end
            end
        end
    end
    return(graph)
end

# Generate a deep copy of the adjacency matrix.
#
# @param graph The given graph.
#
# @return A copy of the adjacency matrix.
function copyAdjacencyMatrix(graph::GraphType)
    adj = zeros(graph.numberOfVertices, graph.numberOfVertices)
    for vertex1 = 1:graph.numberOfVertices
        for vertex2 = 1:graph.numberOfVertices
            adj[vertex1, vertex2] = graph.adjacencyMatrix[vertex1, vertex2]
        end
    end
    return(adj)
end

# Calculate the shortest paths for all vertex pairs.
#
# @param graph The given graph.
#
# @return a numberOfVertices x numberOfVertices Matrix containing the pairwise
#         shortest paths
function allPairShortestPath(graph::GraphType)
    d = copyAdjacencyMatrix(graph)
    for index1 = 1:graph.numberOfVertices
        for index2 = 1:graph.numberOfVertices
            for index3 = 1:graph.numberOfVertices
                d[index2, index3] = min(d[index2, index3], d[index2, index1] + d[index1, index3])
            end
        end
    end

    return(d)
end

end
