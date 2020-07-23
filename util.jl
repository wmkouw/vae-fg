"""Utility functions"""

using Flux: chunk


function convert_to_image(x, y_size)
    "Reshape a mini-batch to an image collage"

    # Map to [0,1] interval
    x = sigmoid.(x)

    # Cut to chunks
    chunks = chunk(x, y_size)

    # Reshape
    img = permutedims(vcat(reshape.(chunks, 28, :)...), (2, 1))

    return img
end
