# fitTest.jl
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

include("fit.jl")

module FitTest

using Fit

# Test the pow-function.
d = 1.0:10.0
k = 0.3
C = 1.4
lambda = 0.5
powTest = pow(d, [k, C, lambda])

if (powTest != k.*(C./d).^(d.*lambda))
    println("powTest failed")
end

# Test the normal-function.
mu = 2.5
sigma = 1.4
normalTest = normal(d, [mu, sigma])

if (normalTest != 1.0 ./ sqrt.(2.0 .* pi .* sigma.^2) .* exp.(-(d-mu).^2 ./ (2.0 .* sigma.^2)))
    println("normalTest failed")
end

# Test the poisson-function.
d = 1.0:10.0
lambda = 0.5
poissonTest = poisson(d, [lambda])

if (poissonTest != lambda .^ d ./ factorial.(d) .* exp.(- lambda))
    println("poissonTest failed")
end

println("FitTest done")
end
