import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import numpy as np
import math

class Particle:
    density = 0
    pressure = 0
    velocity = np.zeros(2)
    position = np.zeros(2)
    type = 0
    def __init__(self, position, velocity, density, mass, type):
        self.position = position
        self.velocity = velocity
        self.density = density
        self.type = type
def make_particles():
    length =1*1000 # meters
    dp = 0.008*1000 # meters
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
            # else:
            if 0<(x1+(dp*i))<0.1 and 0<(y1+(dp*j))<0.1:
                velocity[count] = np.array([1,0])
            count += 1
    
    h = 0.02 *1000
    # h = 0.1
    # h = math.sqrt(3*(dp**2))
    kh = h*2
    return positions, velocity, density, mass, type, kh, h

def plot_NN_grp(positions, NN_idx, title, mid):
    plt.clf()
    plt.scatter(positions[NN_idx[mid][:],0], positions[NN_idx[mid][:],1], s=2)
    # plt.xlim(-0.7, 0.7)
    # plt.ylim(-0.7, 0.7)
    plt.savefig(f'NN_grp_{title}.png')

def plot_velocity_vec(positions, velocity, title): 
    vel_mag = velocity[:,0]**2 + velocity[:,1]**2
    plt.quiver(positions[:,0], positions[:,1], velocity[:,0], velocity[:,1], scale=0.3, scale_units='xy')
    plt.savefig(f'velocity_vec_{title}.png')

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='viridis', s=2)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.savefig(f'{title}.png')

def calc_density(positions, mass, kh, NN_idx, Eta, den_orig, type):
    density = np.zeros(len(positions))
    results = np.zeros(len(positions))
    pool = mp.Pool(mp.cpu_count())
    args = [(i, positions, mass, kh, NN_idx, Eta, den_orig, type) for i in range(len(positions))]
    results = pool.map(calculate_density_for_particle, args)
    pool.close()
    pool.join()
    for i, den in results:
        density[i] = den
    return density

def calculate_density_for_particle(arg):
    i, positions, mass, kh, NN_idx, Eta, density, type = arg
    if type[i] == 0:
        return i, density[i]
    den = 0.0
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance < kh and distance > 0.0:
            # weight = 1
            weight = poly6(kh, distance)
            # weight = ker_w(kh, distance)
            den += weight
            # den += mass * weight
    return i, den

def get_fac_cs(h1):
    # get the kernel normalizing factor
    fac = 10/(7 * np.pi)
    fac = fac * h1 * h1
    return fac

# Cubic spline kernel
def ker_cs(kh,distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = get_fac_cs(h1)
    tmp2 = 2. - q
    if (q > 2.0):
        val = 0.0
    elif (q > 1.0):
        val = 0.25 * tmp2 * tmp2 * tmp2
    else:
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac

def poly6(kh, distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = 4/(np.pi * h**8)
    temp = h**2 - distance**2
    if 0< distance < h:
        return fac * (temp)**3
    else:
        return 0

def get_fac_w(h1):
    # get the kernel normalizing factor
    fac = 7/(4*np.pi)
    fac = fac * h1 * h1
    return fac

def ker_w(kh,distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    temp = 1 - 0.5 * q
    fac = get_fac_w(h1)
    if q <= 2 and q>0:
        return (fac * temp * temp * temp *temp *(2*q +1))
    else:
        return 0

def plot_error_hist(density, den_calc):
    error = density - den_calc
    plt.clf()
    plt.hist(error, bins=100)
    plt.yscale('log')
    plt.savefig('error_hist.png')

def box_plot(error):
    plt.clf()
    plt.boxplot(error, showfliers=False)
    plt.savefig('box_plot.png')

def vol_fluid(mass, density, type):
    vol = 0
    for i in range(len(density)):
        if type[i] == 1:
            vol += mass/density[i]
    return vol

def check_kernel_summation_for_particle(arg):
    i, positions, kh, NN_idx,mass, density= arg
    den = 0.0
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        weight = poly6(kh, distance)
        # weight = ker_w(kh, distance)
        den += (mass/density[i])*weight *distance/(distance+1e-20)
    return i, den

def check_kernel_summation(positions, kh, NN_idx, mass, density):
    # pool = mp.Pool(mp.cpu_count())
    args = [(i, positions, kh, NN_idx, mass, density) for i in range(len(positions))]
    # results = pool.map(check_kernel_summation_for_particle, args)
    results = [check_kernel_summation_for_particle(arg) for arg in args]

    # pool.close()
    # pool.join()
    summatiion = np.zeros(len(positions))
    for i, den in results:
        summatiion[i] = den
    return summatiion

def main():
    pos, vel, density, mass, type, kh, h = make_particles()
    Eta = 1e-20
    radius_ = h*8
    plot_velocity_vec(pos, vel, 'start')
    plot_prop(pos, type, 'type')
    nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)
    NN_idx = nbrs.radius_neighbors(pos)[1]
    print(f'average number of neighbors: {np.mean([len(idx) for idx in NN_idx])}')
    # for i in range(len(NN_idx[1])):
    #     print(pos[NN_idx[1][i],1])
    # ker_calc = check_kernel_summation(pos, kh, NN_idx, mass, density)
    # plot_prop(pos, ker_calc, 'ker_calc')
    mid_idx = int(len(den_calc)/2)
    plot_NN_grp(pos, NN_idx[mid_idx], 'NN', mid_idx)
    list_kh = np.linspace(h, radius_, 1000)
    for i, kh_ in enumerate(list_kh):
        den_calc = calc_density(pos, mass, kh_, NN_idx, Eta, density, type)
        plot_prop(pos, den_calc-density, f'density_diff_{kh_/h}')
    # climin = np.min(den_calc)
    # climax = np.max(den_calc)
    # plot_prop(pos, den_calc, 'density_calc')
    # # plot_prop(pos, den_calc, 'density_calc', climax, climin)
    # plot_prop(pos, density, 'density', climax, climin)
    # plot_error_hist(density, den_calc)
    # box_plot(density-den_calc)
    # vol_pre = vol_fluid(mass, density, type)
    # vol_pos = vol_fluid(mass, den_calc, type)
    # print(f'differnece in volume: {vol_pre-vol_pos}')
    # print(f'% differnece in volume: {((vol_pre-vol_pos)/vol_pre)*100}')



if __name__ == '__main__':
    main()