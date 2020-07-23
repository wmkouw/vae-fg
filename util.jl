"""Utility functions"""

using Flux: chunk


function convert_to_image(X::Array{Float32,2}, num_chunks; rgb=false, patch_size=28)
    "Reshape a mini-batch to an image collage"

    # Check mini-batch size
    batch_size = size(X,2)

    # Map to [0,1] interval
    X = sigmoid.(X)

    if rgb

        # Patch size
        imsz = patch_size*patch_size

        # Split by color channel
        XR = X[0*imsz+1:1*imsz,:]
        XG = X[1*imsz+1:2*imsz,:]
        XB = X[2*imsz+1:3*imsz,:]

        # Cut to chunks
        chunksR = chunk(XR, num_chunks)
        chunksG = chunk(XG, num_chunks)
        chunksB = chunk(XB, num_chunks)

        # Reshape into images
        imagesR = vcat(reshape.(chunksR, patch_size, :)...)
        imagesG = vcat(reshape.(chunksG, patch_size, :)...)
        imagesB = vcat(reshape.(chunksB, patch_size, :)...)

        # Join channels
        images = cat(imagesR, imagesG, imagesB, dims=3)

        # Transpose images
        return permutedims(images, (2, 1, 3))

    else

        # Cut to chunks
        chunks = chunk(X, num_chunks)

        # Reshape into images
        images = vcat(reshape.(chunks, patch_size, :)...)

        # Transpose images
        return permutedims(images, (2, 1))

    end
end
