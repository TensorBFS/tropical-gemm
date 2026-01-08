//! Shortest path computation using tropical matrix multiplication.
//!
//! This example demonstrates how to compute all-pairs shortest paths
//! using TropicalMinPlus matrix multiplication.

use tropical_gemm::prelude::*;

fn main() {
    println!("All-Pairs Shortest Path using Tropical GEMM\n");

    // Create a weighted directed graph with 5 nodes
    // Using f64::INFINITY for no direct edge
    let inf = f64::INFINITY;

    // Distance matrix: dist[i][j] = direct edge weight from i to j
    #[rustfmt::skip]
    let dist = vec![
        0.0, 3.0, inf, 7.0, inf,   // from node 0
        inf, 0.0, 2.0, inf, inf,   // from node 1
        inf, inf, 0.0, 1.0, 5.0,   // from node 2
        inf, inf, inf, 0.0, 2.0,   // from node 3
        inf, inf, inf, inf, 0.0,   // from node 4
    ];

    println!("Initial distance matrix (direct edges):");
    print_matrix(&dist, 5);

    // Compute powers of the distance matrix to find shortest paths
    // D^2 gives shortest 2-hop paths
    // D^4 = D^2 * D^2 gives shortest 4-hop paths
    // For a graph with n nodes, D^(n-1) gives all shortest paths

    let n = 5;

    // D^2
    let d2 = tropical_matmul::<TropicalMinPlus<f64>>(&dist, n, n, &dist, n);
    let d2_values: Vec<f64> = d2.iter().map(|x| x.value()).collect();

    println!("\nD^2 (shortest paths up to 2 hops):");
    print_matrix(&d2_values, 5);

    // D^4 = D^2 * D^2
    let d4 = tropical_matmul::<TropicalMinPlus<f64>>(&d2_values, n, n, &d2_values, n);
    let d4_values: Vec<f64> = d4.iter().map(|x| x.value()).collect();

    println!("\nD^4 (all shortest paths, n-1=4 iterations sufficient):");
    print_matrix(&d4_values, 5);

    // Verify some paths
    println!("\nShortest path distances:");
    println!("  0 -> 4: {} (path: 0->3->4 = 7+2 = 9)", d4_values[4]);
    println!("  0 -> 2: {} (path: 0->1->2 = 3+2 = 5)", d4_values[2]);
    println!(
        "  1 -> 4: {} (path: 1->2->3->4 = 2+1+2 = 5)",
        d4_values[n + 4]
    );

    // With argmax tracking to find the intermediate nodes
    println!("\n--- Path Reconstruction with Argmax ---\n");

    let result =
        tropical_matmul_with_argmax::<TropicalMinPlus<f64>>(&d2_values, n, n, &d2_values, n);

    println!("For D^4, the argmax indicates which intermediate 2-hop point was used:");
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dist = result.get(i, j).value();
                let via = result.get_argmax(i, j);
                if dist < inf {
                    println!("  {} -> {}: distance = {:.1}, via node {}", i, j, dist, via);
                }
            }
        }
    }
}

fn print_matrix(data: &[f64], n: usize) {
    print!("     ");
    for j in 0..n {
        print!("{:5} ", j);
    }
    println!();
    println!("   {}", "-".repeat(n * 6 + 1));

    for i in 0..n {
        print!("{:2} | ", i);
        for j in 0..n {
            let val = data[i * n + j];
            if val == f64::INFINITY {
                print!("  inf ");
            } else {
                print!("{:5.1} ", val);
            }
        }
        println!();
    }
}
