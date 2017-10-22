# histogram.jl
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

module Histogram

export HistogramType, PDFHistogramType, initHistogram, insertHistogram, calculatePDFHistogram, calculateX

# Define a histogram.
#
# @param from Lower bound of the histogram.
# @param to Upper bound of the histogram.
# @param width The width of the bins (assume the width for all bins is equals).
# @param numberOfBins The number of bins.
# @param bins An array representing the bins of the histogram.
# @param numberOfCounts How often has the insert function been called?
mutable struct HistogramType
    from::Float64
    to::Float64
    width::Float64
    numberOfBins::Int
    bins
    numberOfCounts::Int
end

# Define the PDF of an histogram including the error.
#
# @param from Lower bound of the histogram.
# @param to Upper bound of the histogram.
# @param width The width of the bins (assume the width for all bins is equals).
# @param numberOfBins The number of bins.
# @param pdf An array containing the pdf for each histogram bin.
# @param stderr An array containing the standard error for each histogram bin.
mutable struct PDFHistogramType
    from::Float64
    to::Float64
    width::Float64
    numberOfBins::Int
    pdf
    stderr
end

# Initialize a new histogram.
#
# @param from Lower bound of the histogram.
# @param to Upper bound of the histogram.
# @param width The width of the bins (assume the width for all bins is equals).
#
# @return The new histogram.
function initHistogram(from::Float64, to::Float64, width::Float64)
    if (to < from)
        error("Invalid borders, from = ", from, " to = ", to)
    end
    if (width <= 0)
        error("Invalid width = ", width)
    end
    numberOfBins = Int(floor((to - from) / width))
    return(HistogramType(from, to, width, numberOfBins, zeros(numberOfBins), 0))
end

# Insert the given value into the histogram.
#
# @param histogram The histogram where one wants to insert something.
# @param value The value that has to be inserted.
#
# @return The histogram after inserting.
function insertHistogram(histogram::HistogramType, value::Float64)
    if (value < histogram.from || value > histogram.to)
        error("Invalid given value = ", value, " from = ", histogram.from, " to = ", histogram.to)
    end
    # Increment the correct bin
    histogram.bins[Int(floor((value - histogram.from) / histogram.width)) + 1] += 1
    # Update number of counts
    histogram.numberOfCounts += 1
    return(histogram)
end

# Calculate the PDF of the given histogram.
#
# @param histogram The histogram where one wants to generate the PDF.
#
# @return A PDFHistogramType object.
function calculatePDFHistogram(histogram::HistogramType)
    histogramPDF = PDFHistogramType(histogram.from, histogram.to, histogram.width, histogram.numberOfBins, zeros(histogram.numberOfBins), zeros(histogram.numberOfBins))

    auxConst = histogram.numberOfCounts - 1.0 # constant value
    for index = 1:histogram.numberOfBins
        # calculate the pdf
        histogramPDF.pdf[index] = histogram.bins[index] / histogram.numberOfCounts
        # calculate the standard error of the current index
        histogramPDF.stderr[index] = sqrt((histogramPDF.pdf[index] * (1.0 - histogramPDF.pdf[index])) / auxConst)
    end
    return(histogramPDF)
end

# Calculate the x value of the given index.
#
# @param histogram The given histogram.
# @param index The bin index which has to be converted into a x value.
#
# @return The x-value for the index of the given histogram.
function calculateX(histogram, index)
    return((index - 0.5) * histogram.width + histogram.from)
end

end
