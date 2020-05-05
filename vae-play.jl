using Flux
using BSON
using MLDatasets
using Flux.Optimise: update!, train!
using Flux.Data: DataLoader
using Flux: logitbinarycrossentropy
using ProgressMeter: Progress, next!
using Plots
using PyPlot

# Load data sets
X,Y = MLDatasets.MNIST.traindata(Float32)

# Image to vector
X = reshape(X, 28^2, :)

# Data shape
D,N = size(X)

# Start DataLoader
data = DataLoader(X,Y, batchsize=256, shuffle=true)


function encoder(; input_dim=2, hidden_dim=3, latent_dim=1, nonlinearity=tanh)
    "Mapping from observed space to latent space"

    # Map input to hidden layer
    h = Dense(input_dim, hidden_dim, nonlinearity)

    # Map hidden layer activity to mean and log-variance
    μ = Chain(h, Dense(hidden_dim, latent_dim, nonlinearity))
    logσ = Chain(h, Dense(hidden_dim, latent_dim, nonlinearity))

    return μ, logσ
end

function decoder(; latent_dim=1, hidden_dim=3, input_dim=2, nonlinearity=tanh)
   "Mapping from latent space to observed space"

    # Latent to hidden
    h = Dense(latent_dim, hidden_dim, nonlinearity)

    # Hidden to input
    return Chain(h, Dense(hidden_dim, input_dim, nonlinearity))
end

function reconstruct(x, encoder, decoder)
   "Apply encoder and decoder to data"

    # Number of samples
    input_dim, num_samples = size(x)

    # Encode samples
    μ = encoder[1](x)
    logσ = encoder[2](x)

    # Dimensionality of latent space
    latent_dim = size(μ,1)

    # Generate samples in latent space
    z = μ + randn(Float32, (latent_dim, num_samples)) .* exp.(logσ)

    # Decode generated samples
    x_hat = decoder(z)

    return μ, logσ, x_hat
end

function loss(x, encoder, decoder; λ=0.1)
    "Loss layer"

    # Encode and decode data
    μ, logσ, x_hat = reconstruct(x, encoder, decoder)

    # KL-divergence
    KL = 0.5 * sum(@. (exp(2. *logσ) + μ^2 -1. - 2. *logσ)) / N

    # Reconstruction error
    logp_x_z = -sum(logitbinarycrossentropy.(x_hat, x)) / N

    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))

    return -logp_x_z + KL + reg
end

# Training parameters
num_epochs = 10

# Define optimizer
learning_rate = 1e-3
opt = ADAM(learning_rate)

# Define encoder and decoder
enc = encoder(input_dim=784, hidden_dim=500, latent_dim=2)
dec = decoder(latent_dim=2, hidden_dim=500, input_dim=784)

# Extract parameters
ps = Flux.params(enc[1], enc[2], dec)

# Create output directory if not present
!ispath("output") && mkpath("output")

@info "Start Training, total $(num_epochs) epochs"
for epoch in 1:num_epochs

    # Report progress
    @info "Epoch $(epoch)"
    progress = Progress(length(data))

    # Iterate over data
    for (x,y) in data

        # Define gradient function
        gs = gradient(ps) do
            training_loss = loss(x, enc, dec)
          return training_loss
        end

        # Update params
        Flux.update!(opt, ps, gs)

        # Update progress meter
        next!(progress; showvalues=[(:loss, loss(x, enc, dec))])
    end
end

# Save model
BSON.@save "output/model.bson" enc dec ps

# Take first batch of images
x = first(data)[1]

# Reconstruct images
_, _, z = reconstruct(x, enc, dec)

for i = 1:16

    # Visualize images
    x_i = x[:,i] .- minimum(x[:,i])
    x_i = reshape(x_i ./ maximum(x_i), (28,28))'
    PyPlot.imshow(x_i, vmin=0.0, vmax=1.0)
    PyPlot.savefig("output/x_"*string(i)*".png")

    # Visualize reconstructions of images
    z_i = z[:,1] .- minimum(z[:,1])
    z_i = reshape(z_i ./ maximum(z_i), (28,28))'
    PyPlot.imshow(z_i, vmin=0.0, vmax=1.0)
    PyPlot.savefig("output/z_"*string(i)*".png")
end
