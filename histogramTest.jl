# histogramTest.jl
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

include("histogram.jl")

module HistogramTest

using Histogram

# Test initHistogram.
histogram1 = initHistogram(0.0, 10.0, 0.5)

if (histogram1.from != 0.0 && histogram1.to != 10.0 && histogram1.width != 0.5 &&
    histogram1.numberOfBins != 20 && histogram1.bins != zero(20) &&
    histogram1.numberOfCounts != 0)
    println("Test initHistogram failed")
end

# Test insertHistogram.
histogram1 = insertHistogram(histogram1, 5.0)
histogram1 = insertHistogram(histogram1, 3.2)
histogram1 = insertHistogram(histogram1, 5.2)
histogram1 = insertHistogram(histogram1, 7.8)
histogram1 = insertHistogram(histogram1, 0.0)
histogram1 = insertHistogram(histogram1, 9.9)

if (histogram1.numberOfCounts != 6 ||
    histogram1.bins != [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    println("Test insertHistogram failed")
end

# Test calculatePDFHistogram.
# auxiliary pdf and error of histogram1
histPDF = [1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 2.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 1.0 / 6.0]
a = sqrt((1.0 / 6.0) * (1.0 - 1.0 / 6.0) / 5.0)
b = sqrt((2.0 / 6.0) * (1.0 - 2.0 / 6.0) / 5.0)
histERR = [a, 0.0, 0.0, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a]

histogram1PDF = calculatePDFHistogram(histogram1)

if (histogram1PDF.pdf != histPDF || histogram1PDF.stderr != histERR)
    println("Test calculatePDFHistogram failed")
end

# Test calculateX.
resultIndices = [0.25 0.75 1.25 1.75 2.25 2.75 3.25 3.75 4.25 4.75 5.25 5.75 6.25 6.75 7.25 7.75 8.25 8.75 9.25 9.75]

for index=1:20
    if (calculateX(histogram1, index) != resultIndices[index])
        println("Test calculateX failed")
    end
    if (calculateX(histogram1PDF, index) != resultIndices[index])
        println("Test calculateX failed")
    end
end

println("HistogramTest done")
end
