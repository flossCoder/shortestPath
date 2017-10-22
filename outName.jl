# outName.jl
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

module OutName

export outName

# Calculate the out name, which is required to save the results of the
# simulation in a meaningfull way.
#
# @param name Give a short name of the choosen graph generation method.
# @param numberOfVertices Contains the number of verticises of the graph.
# @param step The simulation step that has to be saved.
# @param params Parameter array to generate the graph.
#
# @return The name snippet required to save the results.
function outName(name::String, numberOfVertices::Int, step::Int, params)
    out = string(name, "_", numberOfVertices, "_", step)
    for param in params
        if (Int(ceil(param)) == param)
            out = string(out, "_", Int(ceil(param)))
        else
            out = string(out, "_", param)
        end
    end
    return(out)
end

end
