# <!-- https://www.sciencedirect.com/science/article/pii/S0021999106000295 -->
import numpy as np
import sklearn.neighbors as NN
import multiprocessing as mp
import matplotlib.pyplot as plt

def define_constnats(case, dim = 2):
    #  Example usage
    if case == '2d_disc':
        # 2d - discs
        dim = 2
        h = 0.008485
        kh = h * 2
        dp = 0.005
        mass = 0.025
        file = "../../input_data/data_2d/2d_disc/2d_disc_0014.csv"
    elif case == '2d':
        # 2d
        dim = 2
        h = 0.02
        kh = h *2
        dp = 0.008
        mass = 0.064
        file = "../../input_data/data_2d/out_0099.csv"
    elif case == '3d':
        # 3D
        dim = 3
        h = 0.008660
        kh = h *2
        dp = 0.005
        mass = 0.000125
        file = "./data_3d/Out_0149.csv"
    Eta = dp * 1e-6
    return dim, h, kh, dp, Eta, mass, file

def read_data(file):
    data = np.loadtxt(file, delimiter=";", skiprows=4, usecols=(0,1,2,3,4,5,6,7,8,9))
    # pos_x = data[:, 0]
    # pos_y = data[:, 1]
    # pos_z = data[:, 2]
    positions = np.column_stack((data[:,0], data[:,1], data[:,2]))
    idp = data[:, 4]
    # vel_x = data[:, 4]
    # vel_y = data[:, 5]
    # vel_z = data[:, 6]
    # vel_x, vel_z = rotational_vel(positions)
    velocities = np.column_stack((data[:,4], data[:,5], data[:,6]))
    density = data[:,7]
    # density = (density * 0) +1000
    pressure = data[:,8]
    return positions, velocities, density, pressure, idp

def calculate_velocity_divergence(dim, positions, velocities, density, density_sqr, mass, kh, NN_idx, Eta):
    num_particles = len(positions)
    divergence = np.zeros(num_particles)
    results = np.zeros(num_particles)
    pool = mp.Pool(mp.cpu_count())
    args = [(dim, i, positions, velocities, density, density_sqr, mass, kh, NN_idx, Eta) for i in range(num_particles)]
    results = pool.map(calculate_divergence_for_particle, args)
    # for arg in args:
    #     i, div = calculate_divergence_for_particle(arg)
    #     results[i] = div
    pool.close()
    pool.join()
    for i, div in results:
        divergence[i] = div
    
    return divergence

def calculate_divergence_for_particle(args):
    dim, i, positions, velocities, density, density_sqr, mass, kh, NN_idx, Eta = args
    div = 0.0
    
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        
        if distance < kh and distance > 0.0:
            weight = 1
            # weight = gradient(dim, r_ij, distance, kh)
            # temp = np.dot((velocities[j]/density_sqr[j]) + (velocities[i]/density_sqr[i]), weight)
            temp = np.dot(velocities[j] - velocities[i], weight)
            temp = temp * mass* distance/ (distance + Eta)
            div +=  temp
    # div = div * density[i]
    div = div / density[i]
    
    return i, div


def calc_density(dim, positions, mass, kh, NN_idx, Eta):
    density = np.zeros(len(positions))
    results = np.zeros(len(positions))
    pool = mp.Pool(mp.cpu_count())
    args = [(dim, i, positions, mass, kh, NN_idx, Eta) for i in range(len(positions))]
    results = pool.map(calculate_density_for_particle, args)
    pool.close()
    pool.join()
    for i, den in results:
        density[i] = den
    return density

def calculate_density_for_particle(arg):
    dim, i, positions, mass, kh, NN_idx, Eta = arg
    den = 0.0
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance < kh and distance > 0.0:
            # weight = 1
            weight = ker_w(kh, dim, distance)
            den += mass * weight
    return i, den

def get_fac_cs(h1, dim):
    # get the kernel normalizing factor
    if dim == 1:
        fac = 2/3
        fac = fac * h1
        return fac
    elif dim == 2:
        fac = 10/(7 * np.pi)
        fac = fac * h1 * h1
        return fac
    elif dim == 3:
        fac = 1/(np.pi)
        fac = fac * h1 * h1 * h1
        return fac

# Cubic spline kernel
def ker_cs(kh, dim, distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = get_fac_cs(h1, dim)
    tmp2 = 2. - q
    if (q > 2.0):
        val = 0.0
    elif (q > 1.0):
        val = 0.25 * tmp2 * tmp2 * tmp2
    else:
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac

def get_fac_w(h1, dim):
    # get the kernel normalizing factor
    if dim == 1:
        fac = 2
        fac = fac * h1
        return fac
    elif dim == 2:
        fac = 7/(4*np.pi)
        fac = fac * h1 * h1
        return fac
    elif dim == 3:
        fac = 21/(16 * np.pi)
        fac = fac * h1 * h1 * h1
        return fac
def ker_w(kh, dim, distance):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    temp = 1 - 0.5 * q
    fac = get_fac_w(h1, dim)
    if q <2 and q>0:
        return (fac * temp * temp * temp *temp *(2*q +1))
    else:
        return 0
    

def plot_comp(positions, data, parameter):
    plt.clf()
    plt.scatter(positions[:, 0], positions[:, 2], c=data)
    plt.colorbar()
    plt.savefig(f"output_{parameter}.png")

def main():
    dim, h, kh, dp, Eta, mass, file = define_constnats('2d')
    # dim, h, kh, dp, Eta, mass, file = define_constnats('2d_disc')
    positions, velocities, density, pressure, idp = read_data(file)
    nbrs = NN.NearestNeighbors(radius=kh, algorithm='kd_tree').fit(positions)
    NN_idx = nbrs.radius_neighbors(positions)[1]
    print(f"Average number of neighbors: {np.mean([len(idx) for idx in NN_idx])}")
    denity_sqr = density ** 2
    den_calc = calc_density(dim, positions, mass, kh, NN_idx, Eta)
    density_diff = den_calc - density
    max_diff = np.max(np.abs(density_diff))
    print(f"Max difference in density: {max_diff}")
    avg_diff = np.mean(np.abs(density_diff))
    print(f"Average difference in density: {avg_diff}")
    plot_comp(positions, density_diff, 'density_diff')
    plot_comp(positions, den_calc, 'density_calc')
    plot_comp(positions, density, 'density')




if __name__ == '__main__':
    main()