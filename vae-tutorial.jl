using Flux
using BSON
using MLDatasets
using Flux.Optimise: update!, train!
using Flux.Data: DataLoader
using Flux: logitbinarycrossentropy
using CUDAapi: has_cuda_gpu
using ProgressMeter: Progress, next!
using Plots
using PyPlot
include("util.jl")


# Experimental parameters
batch_size = 256
input_dim = 28^2
hidden_dim = 500
latent_dim = 2
num_epochs = 10
learning_rate = 1e-3
regularization_param = 0.01

# GPU config
if has_cuda_gpu()
    device = gpu
    @info "Training on GPU"
else
    device = cpu
    @info "Training on CPU"
end

# Load data sets
X,Y = MLDatasets.MNIST.traindata(Float32)

# Image to vector
X = reshape(X, input_dim, :)

# Data shape
D,N = size(X)

# Start DataLoader
data = DataLoader(X,Y, batchsize=batch_size, shuffle=true)

struct Encoder
    linear
    μ
    logσ
    Encoder(; input_dim=1, latent_dim=1, hidden_dim=1, nonlinearity=tanh, device=cpu) = new(
        Dense(input_dim, hidden_dim, nonlinearity) |> device,   # linear
        Dense(hidden_dim, latent_dim) |> device,        # μ
        Dense(hidden_dim, latent_dim) |> device,        # logσ
    )
end

function (encoder::Encoder)(x)

    # Map input to hidden layer
    h = encoder.linear(x)

    # Return mean and log-sigma of encoder
    return encoder.μ(h), encoder.logσ(h)
end

function Decoder(; latent_dim=1, hidden_dim=1, input_dim=1, nonlinearity=tanh, device=cpu)
   "Mapping from latent space to observed space"

    dec = Chain(

        # Latent to hidden
        Dense(latent_dim, hidden_dim, nonlinearity),

        # Hidden to input
        Dense(hidden_dim, input_dim, nonlinearity)

    ) |> device

    return dec
end

function reconstruct(x, encoder, decoder; device=cpu)
   "Apply encoder and decoder to data"

    # Number of samples
    input_dim, num_samples = size(x)

    # Encode samples
    μ, logσ = encoder(x)

    # Dimensionality of latent space
    latent_dim = size(μ,1)

    # Generate samples in latent space and decode
    z = decoder(μ + device(randn(Float32, (latent_dim, num_samples))) .* exp.(logσ))

    return μ, logσ, z
end

function loss(encoder, decoder, x; regularization_param=0.01, device=cpu)
    "Loss layer"

    # Encode and decode data
    μ, logσ, z = reconstruct(x, encoder, decoder, device=device)

    # KL-divergence
    divergence = 0.5 * sum(@. (exp(2. *logσ) + μ^2 -1. - 2. *logσ))

    # Reconstruction error
    logp_x_z = -sum(logitbinarycrossentropy.(z, x))

    # regularization
    regularization = regularization_param * sum(x->sum(x.^2), Flux.params(decoder))

    return -logp_x_z + divergence + regularization
end

# Define optimizer
opt = ADAM(learning_rate)

# Define encoder and decoder
enc = Encoder(input_dim=784, hidden_dim=500, latent_dim=2, device=device)
dec = Decoder(latent_dim=2, hidden_dim=500, input_dim=784, device=device)

# Extract parameters
ps = Flux.params(enc.linear, enc.μ, enc.logσ, dec)

# Create output directory if not present
!ispath("output") && mkpath("output")

@info "Start Training, total $(num_epochs) epochs"
for epoch in 1:num_epochs

    # Report progress
    @info "Epoch $(epoch)"
    progress = Progress(length(data))

    # Iterate over data
    for (x,_) in data

        # Define gradient function
        training_loss, back = Flux.pullback(ps) do
            loss(enc, dec, x |> device, device=device)
        end
        grad = back(1.)

        # Update params
        Flux.Optimise.update!(opt, ps, grad)

        # Update progress meter
        next!(progress; showvalues=[(:loss, training_loss)])
    end
end

# Save model
BSON.@save "output/model.bson" enc dec ps

# Take first batch of images and map to image collage
x = first(data)[1]
img_x = convert_to_image(x, 16)
PyPlot.imshow(img_x, vmin=0.0, vmax=1.0, cmap="gray")
PyPlot.savefig("output/collage_x.png")

# Reconstruct images and map to collage
_, _, z = reconstruct(x, enc, dec)
img_z = convert_to_image(z, 16)
PyPlot.imshow(img_z, vmin=0.0, vmax=1.0, cmap="gray")
PyPlot.savefig("output/collage_z.png")
