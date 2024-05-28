import numpy as np
import sklearn.neighbors as NN
from PPE_main import calculate_velocity_divergence as ck
from PPE_main import check_non_divergent as ck2
import multiprocessing as mp

def find_neighbors(positions, h):
    num_particles = positions.shape[0]
    neighbors = []
    for i in range(num_particles):
        neighbors.append([j for j in range(num_particles) if np.linalg.norm(positions[i] - positions[j]) < h and i != j])
    return neighbors

def calculate_density(positions, mass, h):
    num_particles = positions.shape[0]
    density = np.zeros(num_particles)
    for i in range(num_particles):
        for j in range(num_particles):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            if r_ij < h:
                density[i] += mass * (h - r_ij) / h
    return density

def calculate_pressure(density, k=1.0):
    return k * (density - 1000)

def calculate_forces(positions, velocities, pressure, density, mass, h, dt):
    num_particles = positions.shape[0]
    forces = np.zeros((num_particles, 3))
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                r_ij = positions[i] - positions[j]
                distance = np.linalg.norm(r_ij)
                if distance < h:
                    weight = (h - distance) / h
                    pressure_term = -mass * (pressure[i] + pressure[j]) / (2 * density[j]) * weight / (distance + 1e-6)
                    forces[i] += pressure_term * r_ij / distance
    return forces

def update_velocities(velocities, forces, density, dt):
    return velocities + dt * forces / density

def update_positions(positions, velocities, dt):
    return positions + dt * velocities

def calculate_divergence_for_particle(args):
    i, positions, velocities, density, mass, h, NN_idx = args
    div = 0.0
    
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        
        if distance < h and distance > 0.0:
            weight = (h - distance) / h
            div += np.dot(velocities[i] - velocities[j], r_ij) / (distance + 1e-6)
    
    return i, div

def calculate_velocity_divergence(positions, velocities, density, mass, h, NN_idx):
    """
    Calculate the velocity divergence for each particle in parallel.
    
    Parameters:
    - positions: array of particle positions (N x 3)
    - velocities: array of particle velocities (N x 3)
    - density: array of particle densities (N)
    - mass: mass of each particle (scalar)
    - h: smoothing length
    - NN_idx: list of nearest neighbors for each particle
    
    Returns:
    - divergence: array of velocity divergences for each particle (N)
    """
    num_particles = len(positions)
    divergence = np.zeros(num_particles)
    
    # Create a pool of worker processes
    pool = mp.Pool(mp.cpu_count())
    
    # Prepare arguments for each particle
    args = [(i, positions, velocities, density, mass, h, NN_idx) for i in range(num_particles)]
    
    # Compute divergence in parallel
    results = pool.map(calculate_divergence_for_particle, args)
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    
    # Collect the results
    for i, div in results:
        divergence[i] = div
    
    return divergence

def update_pressure(args):
    i, positions, velocities, density, pressure, mass, dt, h, NN_idx = args
    pressure_gradient = np.zeros(3)
    divergence = 0.0

    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        
        if distance < h and distance > 0.0:
            weight = (h - distance) / h
            pressure_gradient += weight * (pressure[i] - pressure[j]) * r_ij / (distance + 1e-6)
            divergence += np.dot(velocities[i] - velocities[j], r_ij) / (distance + 1e-6)

    new_pressure = (pressure[i] - (mass / density[i]) * (divergence / dt) * h * h) / (1 + h * h)
    return i, new_pressure

def update_velocity(args):
    i, positions, velocities, density, pressure, dt, h, NN_idx = args
    pressure_gradient = np.zeros(3)

    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        
        if distance < h and distance > 0.0:
            weight = (h - distance) / h
            pressure_gradient += weight * (pressure[i] - pressure[j]) * r_ij / (distance + 1e-6)

    new_velocity = velocities[i] - dt * pressure_gradient / density[i]
    return i, new_velocity

def solve_pressure_poisson(positions, velocities, density, pressure, mass, dt, h, NN_idx, tolerance=1e-6, max_iterations=1000):
    """
    Solves the Pressure Poisson Equation to enforce a divergence-free velocity field.
    
    Parameters:
    - positions: array of particle positions (N x 3)
    - velocities: array of particle velocities (N x 3)
    - density: array of particle densities (N)
    - pressure: array of particle pressures (N)
    - mass: mass of each particle (scalar)
    - dt: time step
    - h: smoothing length
    - NN_idx: list of nearest neighbors for each particle
    - tolerance: convergence tolerance for the iterative solver
    - max_iterations: maximum number of iterations for the solver
    
    Returns:
    - Updated pressures and velocities
    """
    num_particles = len(positions)
    
    pool = mp.Pool(mp.cpu_count())
    
    # Iteratively solve the Pressure Poisson Equation
    for iteration in range(max_iterations):
        # print(f"Iteration {iteration}")
        pressure_old = pressure.copy()

        # Prepare arguments for each particle
        pressure_args = [(i, positions, velocities, density, pressure, mass, dt, h, NN_idx) for i in range(num_particles)]
        
        # Compute pressure in parallel
        pressure_results = pool.map(update_pressure, pressure_args)
        
        # Collect the results
        for i, new_pressure in pressure_results:
            pressure[i] = new_pressure

        # Check for convergence
        max_error = np.max(np.abs(pressure - pressure_old))
        if max_error < tolerance:
            break

    # Prepare arguments for each particle
    velocity_args = [(i, positions, velocities, density, pressure, dt, h, NN_idx) for i in range(num_particles)]
    
    # Update velocities in parallel
    velocity_results = pool.map(update_velocity, velocity_args)
    
    # Collect the results
    for i, new_velocity in velocity_results:
        velocities[i] = new_velocity

    pool.close()
    pool.join()

    return pressure, velocities

def sph_simulation_step(positions, velocities, mass, h, dt):
    density = calculate_density(positions, mass, h)
    pressure = calculate_pressure(density)
    forces = calculate_forces(positions, velocities, pressure, density, mass, h, dt)
    velocities = update_velocities(velocities, forces, density, dt)
    positions = update_positions(positions, velocities, dt)
    return positions, velocities, density, pressure

# Example usage
num_particles = 100
positions = np.random.rand(num_particles, 3)
velocities = np.random.rand(num_particles, 3) - 0.5  # Initialize with some random velocities
density = np.ones(num_particles) * 1000  # Assuming uniform density of 1000 kg/m^3
pressure = np.zeros(num_particles)
mass = 1.0  # Mass of each particle
dt = 0.01  # Time step
h = 0.1  # Smoothing length
# NN_idx = find_neighbors(positions, h)  # Find nearest neighbors for each particle

kh = 2* h
dim =3
nbrs = NN.NearestNeighbors(radius=kh, algorithm='kd_tree').fit(positions)
NN_idx = nbrs.radius_neighbors(positions)[1]
divergence_post = ck(dim, positions, velocities, density, mass, kh, NN_idx)
# divergence = calculate_velocity_divergence(positions, velocities, density, mass, Kernel, NN_idx)
is_non_divergent, max_divergence_post = ck2(divergence_post)
print(f"Is the flow non-divergent? {is_non_divergent}")
print(f"Maximum divergence: {max_divergence_post}")

pressure, velocities = solve_pressure_poisson(positions, velocities, density, pressure, mass, dt, h, NN_idx)

divergence_post = ck(dim, positions, velocities, density, mass, kh, NN_idx)
# divergence = calculate_velocity_divergence(positions, velocities, density, mass, Kernel, NN_idx)
is_non_divergent, max_divergence_post = ck2(divergence_post)
print(f"Is the flow non-divergent? {is_non_divergent}")
print(f"Maximum divergence: {max_divergence_post}")
# print("Updated Pressures:", pressure)
# print("Updated Velocities:", velocities)
