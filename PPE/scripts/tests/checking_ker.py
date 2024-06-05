import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import numpy as np
import math

def make_particles():
    length =1 # meters
    dp = 0.008 # meters
    # length =1 # meters
    # dp = 0.008 # meters
    boundary_width = dp*40 # meters
    x1 = -length/2 - boundary_width/2
    x2 = length/2 + boundary_width/2
    y1 = -length/2 - boundary_width/2
    y2 = length/2 + boundary_width/2
    resolution = int((length+boundary_width)/dp)
    positions = np.zeros((resolution**2,2))
    density = np.ones(resolution**2)
    density = density * 1000
    velocity = np.zeros((resolution**2,2))
    mass = density[1]*dp**2
    type = np.zeros(resolution**2)
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            positions[count] = np.array([(x1+(dp*i)),(y1+(dp*j))])
            if x1+boundary_width/2<(x1+(dp*i))<x2-dp-boundary_width/2 and y1+boundary_width/2<(y1+(dp*j))<y2-dp-boundary_width/2:
                type[count] = 1
            count += 1
    
    h = 0.02 
    kh = h*2
    temp = ((resolution*resolution/2 ))    
    # temp = temp + ((resolution )*2)
    temp = int(temp)
    return positions, velocity, density, mass, type, kh, h, temp

def poly6(kh, distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    if 0< distance < h:
        return fac * (temp)**3
    else:
        return 0

def check_kernel_summation_for_particle(arg):
    i, positions, kh, NN_idx,= arg
    den = 0.0
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance < kh and distance > 0.0:
            weight = poly6(kh, distance)
            # weight = ker_w(kh, distance)
            den += weight 
            # den += weight *distance/(distance+1e-20)
    return i, den

def check_kernel_summation(positions, kh, NN_idx):
    pool = mp.Pool(mp.cpu_count())
    args = [(i, positions, kh, NN_idx) for i in range(len(positions))]
    results = pool.map(check_kernel_summation_for_particle, args)
    # results = [check_kernel_summation_for_particle(arg) for arg in args]

    pool.close()
    pool.join()
    summatiion = np.zeros(len(positions))
    for i, den in results:
        summatiion[i] = den
    return summatiion

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='viridis', s=2)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.savefig(f'{title}.png')

def plot_NN_grp(positions, NN_idx, title, mid):
    plt.clf()
    plt.scatter(positions[NN_idx[mid][:],0], positions[NN_idx[mid][:],1], s=2)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.savefig(f'NN_grp_{title}.png')


def main():
    pos, vel, density, mass, type, kh, h, mid = make_particles()
    Eta = 1e-20
    nbrs = NN.NearestNeighbors(radius=2*kh, algorithm='kd_tree').fit(pos)
    NN_idx = nbrs.radius_neighbors(pos)[1]
    print(f'average number of neighbors: {np.mean([len(idx) for idx in NN_idx])}')
    h_vals = np.linspace(h, 2*h, 1000)
    ker_vals = np.zeros(len(h_vals))
    plot_NN_grp(pos, NN_idx, 'start', mid)
    for i, h_val in enumerate(h_vals):
        kh_ = 2*h_val
        arg =(mid, pos, kh_, NN_idx)
        _, ker_vals[i] = check_kernel_summation_for_particle(arg)
    
    plt.clf()
    plt.plot(h_vals, ker_vals) 
    plt.savefig('ker_vals.png')

    # ker_calc = check_kernel_summation(pos, kh, NN_idx)
    # plot_prop(pos, ker_calc, 'ker_calc')

if __name__ == "__main__":
    main()