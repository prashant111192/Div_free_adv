import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as NN


def make_particles():
    # Number of particles
    N = 100

    # Number of dimensions
    D = 2

    particles = np.zeros((N, D))
    count =0
    for i in range(10):
        for j in range(10):
            particles[count] = np.array([i, j])
            count += 1
    return particles

def find_nn(particles, radius):
    nn_list = []
    count = 0
    for i in range(0, len(particles)):
        print(f"Finding neighbours for particle {i}")
        nn_local_list = []
        dist = np.linalg.norm(particles - particles[i], axis=1)
        for j in range(len(particles)):
            if dist[j] <= radius and j != i:
                nn_local_list.append(j)
                count = count + 1
                print(f"Particle {j} is a neighbour of particle {i} with a distance of {dist[j]}")
        nn_list.append(nn_local_list)
    
    print(f'Total number of neighbours: {count}')
    return nn_list




def main():

    particle = make_particles()
    nn_list = find_nn(particle, 3)
    nbrs = NN.NearestNeighbors(radius=3, algorithm='kd_tree').fit(particle)
    NN_idx = nbrs.radius_neighbors(particle)[1]
    print(f'total number of neighbors using NN: {np.sum([len(idx) for idx in NN_idx])}')





if __name__ == '__main__':
    main()