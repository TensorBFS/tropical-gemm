#!/usr/bin/env julia
#
# Julia Reference Data Generator for tropical-gemm Tests
#
# This script generates ground-truth test data using TropicalNumbers.jl
# to validate the Rust/Python implementation.
#
# Usage:
#   julia julia_generator.jl
#
# Output:
#   JSON files in subdirectories organized by algebra type and scalar type

using Pkg

# Ensure required packages are available
required_packages = ["TropicalNumbers", "JSON3", "Random"]
for pkg in required_packages
    try
        # Check if package is available in current environment
        if !haskey(Pkg.project().dependencies, pkg)
            println("Installing $pkg...")
            Pkg.add(pkg)
        end
    catch
        # Fallback: try to add if project check fails
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

using TropicalNumbers
using JSON3
using Random

# Set seed for reproducibility
Random.seed!(42)

# Output directory (same as this script)
const OUTPUT_DIR = @__DIR__

# ============================================================================
# Configuration
# ============================================================================

# Algebra configurations: (name, scalar_name, julia_type, scalar_generator)
const ALGEBRA_CONFIGS = [
    # MaxPlus: ⊕ = max, ⊗ = +
    ("maxplus", "f32", Tropical{Float32}, () -> randn(Float32) * 10),
    ("maxplus", "f64", Tropical{Float64}, () -> randn(Float64) * 10),

    # MinPlus: ⊕ = min, ⊗ = +
    ("minplus", "f32", TropicalMinPlus{Float32}, () -> randn(Float32) * 10),
    ("minplus", "f64", TropicalMinPlus{Float64}, () -> randn(Float64) * 10),

    # MaxMul: ⊕ = max, ⊗ = *
    # Use positive values to avoid sign issues with multiplication
    ("maxmul", "f32", TropicalMaxMul{Float32}, () -> abs(randn(Float32)) + 0.1f0),
    ("maxmul", "f64", TropicalMaxMul{Float64}, () -> abs(randn(Float64)) + 0.1),

    # AndOr: ⊕ = OR, ⊗ = AND
    ("andor", "bool", TropicalAndOr, () -> rand(Bool)),

    # CountingTropical: tracks count of optimal paths
    ("counting", "f32", CountingTropical{Float32}, () -> randn(Float32) * 10),
]

# Matrix shapes: (m, k, n) - max 30x30 to keep fixture size small
const SHAPES = [
    (4, 4, 4),       # Tiny square
    (8, 8, 8),       # Small square
    (16, 16, 16),    # Medium square
    (30, 30, 30),    # Largest square
    (12, 24, 18),    # Rectangular
]

# Batch size for batched operations
const BATCH_SIZE = 3

# ============================================================================
# Helper Functions
# ============================================================================

"""
Extract the scalar value from a tropical number.
"""
function extract_value(x::Tropical{T}) where T
    return x.n
end

function extract_value(x::TropicalMinPlus{T}) where T
    return x.n
end

function extract_value(x::TropicalMaxMul{T}) where T
    return x.n
end

function extract_value(x::TropicalAndOr)
    return x.n
end

function extract_value(x::CountingTropical{T}) where T
    # Return as [value, count] pair
    return [x.n, x.c]
end

"""
Convert a matrix of tropical numbers to a nested array of scalar values.
Matrices are stored in row-major order (C-contiguous) for Python/NumPy compatibility.
Julia uses column-major, so we need to transpose.
"""
function to_row_major(mat::AbstractMatrix)
    m, n = size(mat)
    result = Vector{Vector}(undef, m)
    for i in 1:m
        result[i] = [extract_value(mat[i, j]) for j in 1:n]
    end
    return result
end

"""
Generate a random matrix with the given shape and element generator.
"""
function generate_matrix(m::Int, k::Int, ::Type{T}, gen::Function) where T
    data = [gen() for _ in 1:m, _ in 1:k]
    return T.(data)
end

"""
Compute argmax indices for tropical matmul C = A ⊗ B.
For each C[i,j], find the k that achieved the optimal value.
"""
function compute_argmax(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T
    m, k_dim = size(a)
    _, n = size(b)

    argmax_indices = Matrix{Int}(undef, m, n)

    for i in 1:m
        for j in 1:n
            # Compute all k contributions
            best_k = 1
            best_val = a[i, 1] * b[1, j]  # Tropical multiplication

            for k in 2:k_dim
                val = a[i, k] * b[k, j]
                # For tropical semirings, addition is the "max" or "min" operation
                combined = best_val + val
                if extract_value(combined) == extract_value(val)
                    # This k achieved the optimal value
                    best_val = val
                    best_k = k
                elseif extract_value(combined) != extract_value(best_val)
                    # Neither won cleanly, combined is the winner
                    best_val = combined
                end
            end

            # Recompute to find actual argmax (first occurrence)
            best_k = 1
            best_val = a[i, 1] * b[1, j]
            for k in 2:k_dim
                val = a[i, k] * b[k, j]
                if extract_value(val + best_val) != extract_value(best_val)
                    best_val = val
                    best_k = k
                end
            end

            argmax_indices[i, j] = best_k - 1  # 0-indexed for Python
        end
    end

    return argmax_indices
end

"""
Convert argmax matrix to row-major nested array.
"""
function argmax_to_row_major(argmax::AbstractMatrix{Int})
    m, n = size(argmax)
    result = Vector{Vector{Int}}(undef, m)
    for i in 1:m
        result[i] = [argmax[i, j] for j in 1:n]
    end
    return result
end

# ============================================================================
# Dataset Generation
# ============================================================================

"""
Generate a single test case (non-batched).
"""
function generate_test_case(
    algebra_name::String,
    scalar_name::String,
    ::Type{T},
    gen::Function,
    m::Int, k::Int, n::Int;
    with_argmax::Bool = true
) where T
    # Generate random matrices
    a = generate_matrix(m, k, T, gen)
    b = generate_matrix(k, n, T, gen)

    # Compute result using Julia's multiple dispatch
    c = a * b

    # Build result dictionary
    result = Dict{String, Any}(
        "algebra" => algebra_name,
        "scalar" => scalar_name,
        "m" => m,
        "k" => k,
        "n" => n,
        "batch_size" => nothing,
        "a" => to_row_major(a),
        "b" => to_row_major(b),
        "c_expected" => to_row_major(c),
    )

    # Compute argmax if requested (not for CountingTropical)
    if with_argmax && algebra_name != "counting"
        argmax = compute_argmax(a, b)
        result["argmax_expected"] = argmax_to_row_major(argmax)
    end

    return result
end

"""
Generate a batched test case.
"""
function generate_batched_test_case(
    algebra_name::String,
    scalar_name::String,
    ::Type{T},
    gen::Function,
    m::Int, k::Int, n::Int,
    batch_size::Int;
    with_argmax::Bool = true
) where T
    # Generate batch of matrices
    a_batch = [generate_matrix(m, k, T, gen) for _ in 1:batch_size]
    b_batch = [generate_matrix(k, n, T, gen) for _ in 1:batch_size]

    # Compute results
    c_batch = [a_batch[i] * b_batch[i] for i in 1:batch_size]

    # Build result dictionary
    result = Dict{String, Any}(
        "algebra" => algebra_name,
        "scalar" => scalar_name,
        "m" => m,
        "k" => k,
        "n" => n,
        "batch_size" => batch_size,
        "a" => [to_row_major(a) for a in a_batch],
        "b" => [to_row_major(b) for b in b_batch],
        "c_expected" => [to_row_major(c) for c in c_batch],
    )

    # Compute argmax if requested
    if with_argmax && algebra_name != "counting"
        argmax_batch = [compute_argmax(a_batch[i], b_batch[i]) for i in 1:batch_size]
        result["argmax_expected"] = [argmax_to_row_major(am) for am in argmax_batch]
    end

    return result
end

"""
Generate special test case with specific value patterns.
"""
function generate_special_case(
    algebra_name::String,
    scalar_name::String,
    ::Type{T},
    special_type::Symbol,
    m::Int, k::Int, n::Int
) where T
    gen = if special_type == :zeros
        # Sparse zeros for MaxMul edge case
        () -> rand() < 0.3 ? zero(Float32) : abs(randn(Float32)) + 0.1f0
    elseif special_type == :negative
        # All negative values
        () -> -abs(randn(Float32)) - 0.1f0
    elseif special_type == :infinity
        # Mix with infinity values
        () -> rand() < 0.1 ? (rand(Bool) ? Inf32 : -Inf32) : randn(Float32) * 10
    else
        error("Unknown special type: $special_type")
    end

    return generate_test_case(algebra_name, scalar_name, T, gen, m, k, n; with_argmax=true)
end

# ============================================================================
# Main Generation Loop
# ============================================================================

function shape_to_name(m::Int, k::Int, n::Int)
    if m == k == n
        return "square_$m"
    else
        return "rect_$(m)x$(k)x$(n)"
    end
end

function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

function save_json(path::String, data::Dict)
    open(path, "w") do f
        JSON3.write(f, data)  # Compact output, no pretty printing
    end
    println("  Created: $path")
end

function main()
    println("=" ^ 60)
    println("Julia Reference Data Generator for tropical-gemm")
    println("=" ^ 60)
    println()

    total_files = 0

    for (algebra_name, scalar_name, T, gen) in ALGEBRA_CONFIGS
        dir_name = "$(algebra_name)_$(scalar_name)"
        dir_path = joinpath(OUTPUT_DIR, dir_name)
        ensure_dir(dir_path)

        println("Generating $dir_name...")

        # Generate non-batched test cases for each shape
        for (m, k, n) in SHAPES
            shape_name = shape_to_name(m, k, n)

            # Basic test case with argmax
            data = generate_test_case(algebra_name, scalar_name, T, gen, m, k, n)
            save_json(joinpath(dir_path, "$shape_name.json"), data)
            total_files += 1
        end

        # Generate batched test cases (only for smaller sizes to keep file size reasonable)
        for (m, k, n) in [(8, 8, 8), (16, 16, 16)]
            shape_name = shape_to_name(m, k, n)
            data = generate_batched_test_case(algebra_name, scalar_name, T, gen, m, k, n, BATCH_SIZE)
            save_json(joinpath(dir_path, "batched_$(shape_name)_b$(BATCH_SIZE).json"), data)
            total_files += 1
        end

        println()
    end

    # Generate special test cases for key algebras
    println("Generating special cases...")

    special_configs = [
        ("maxmul", "f32", TropicalMaxMul{Float32}, :zeros),
        ("maxplus", "f32", Tropical{Float32}, :negative),
        ("minplus", "f32", TropicalMinPlus{Float32}, :negative),
    ]

    for (algebra_name, scalar_name, T, special_type) in special_configs
        dir_name = "$(algebra_name)_$(scalar_name)"
        dir_path = joinpath(OUTPUT_DIR, dir_name)

        for (m, k, n) in [(8, 8, 8), (16, 16, 16)]
            shape_name = shape_to_name(m, k, n)
            data = generate_special_case(algebra_name, scalar_name, T, special_type, m, k, n)
            save_json(joinpath(dir_path, "special_$(special_type)_$(shape_name).json"), data)
            total_files += 1
        end
    end

    println()
    println("=" ^ 60)
    println("Generation complete!")
    println("Total files created: $total_files")
    println("=" ^ 60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
