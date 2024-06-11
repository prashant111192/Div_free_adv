import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import numpy as np
import math

def make_particles(length =1, boundary_fac = 40, dp=0.008):
    boundary_width = dp*boundary_fac # meters

    x1 = -length/2 - boundary_width/2
    x2 = length/2 + boundary_width/2
    y1 = -length/2 - boundary_width/2
    y2 = length/2 + boundary_width/2

    resolution = int((length+boundary_width)/dp)
    pos = np.zeros((resolution**2,2))
    density = np.ones(resolution**2)
    density = density * 1000
    velocity = np.zeros((resolution**2,2))
    mass = density[1]*dp**2
    p_type = np.zeros(resolution**2)

    count = 0
    for i in range(resolution):
        for j in range(resolution):
            pos[count] = np.array([(x1+(dp*i)),(y1+(dp*j))])
            if x1+boundary_width/2<(x1+(dp*i))<x2-dp-boundary_width/2 and y1+boundary_width/2<(y1+(dp*j))<y2-dp-boundary_width/2:
                p_type[count] = 1 # Fluid particle
                # velocity[count] = np.array([math.sin(pos[count][0])*0.01,math.cos(pos[count][1])*0.01])
                if pos[count][0] > 0 and pos[count][1] > 0 and pos[count][0] < 30*dp and pos[count][1] < 30*dp:
                    velocity[count] = np.array([0.05,0.05])
            count += 1
    
    # h = 0.008660
    h_const = 0.02 
    dp_const = 0.008
    h_fac = h_const/dp_const
    h = h_fac * dp

    kh = h*2
    mid = ((resolution*resolution/2 ))    
    mid = int(mid)
    return pos, velocity, density, mass, p_type, kh, h, mid

def poly6(kh, distance):
    h = kh
    # h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    if 0 < distance <= h:
        return fac * (temp)**3
    else:
        return 0 

def plot_velocity_vec(positions, velocity, title): 
    vel_mag = velocity[:,0]**2 + velocity[:,1]**2
    pos = positions[::25,:]
    vel = velocity[::25,:]
    plt.cla()
    plt.quiver(pos[:,0], pos[:,1], vel[:,0], vel[:,1], scale=1, scale_units='xy')
    # plt.quiver(positions[:,0], positions[:,1], velocity[:,0], velocity[:,1], scale=0.3, scale_units='xy')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'velocity_vec_{title}.png')

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='viridis', s=0.2, alpha=0.5)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{title}.png')

def calc_density(positions, mass, kh, NN_idx, Eta, den_orig, type):
    density = np.zeros(len(positions))
    results = np.zeros(len(positions))
    res_idx = np.zeros(len(positions), dtype=int)
    pool = mp.Pool(6)
    # args = [(i, positions, mass, kh, NN_idx, Eta, den_orig, type) for i in range(10)]
    args = [(i, positions, mass, kh, NN_idx, Eta, den_orig, type) for i in range(len(positions))]
    # for i, arg in enumerate(args):
    #     res_idx[i], results[i] = calculate_density_for_particle(arg)
    
    results = pool.map(calc_density_for_particle, args)
    pool.close()
    pool.join()
    # print(results)
    for i, den in results:
        density[i] = den
    # print('done')
    return density

def calc_density_for_particle(arg):
    i, positions, mass, kh, NN_idx, Eta, density, type = arg
    # if type[i] == 0: # boundary particle
    #     return i, density[i]
    den = 0.0
    for j in NN_idx[i]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance < kh and distance > 0.0:
            # weight = 1
            weight = poly6(kh, distance)
            # weight = ker_w(kh, distance)
            den += weight *mass
            # den += mass * weight
    return i, den

def calc_divergence(pos, vel, mass, kh, NN_idx, Eta, density, p_type):
    
    n_particles = len(density)
    divergence = np.zeros(len(density))
    results = np.zeros(len(density))
    pool = mp.Pool(6)
    args = [(i, pos,vel , density, mass, kh, NN_idx, Eta, p_type) for i in range(n_particles)]
    # for arg in args:
    #     i, div = calc_divergence_part(arg)
    #     divergence[i] = div
    results = pool.map(calc_divergence_part, args)
    pool.close()
    pool.join()
    for i, div in results:
        divergence[i] = div
    return divergence

def gradient_poly6(r_ij, distance, kh):
    h = kh
    h1 = 1/h
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    grad = np.zeros(2)
    if 0 < distance <= h:
        temp_2 = fac * 3*((temp)**2)*(-2*distance) * h1 / distance
        grad = temp_2 * r_ij

        return grad
    else:
        return grad 


def calc_divergence_part(arg):
    i, pos, vel, density, mass, kh, NN_idx, Eta, p_type = arg

    div = 0.0
    
    if p_type[i] !=0:
        for j in NN_idx[i]:
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
        
            if distance < kh and distance > 0.0:
                weight = gradient_poly6(r_ij, distance, kh)
                # temp = np.dot((vel[j]/density_sqr[j]) + (vel[i]/density_sqr[i]), weight)
                temp = np.dot(vel[j] - vel[i], weight)
                temp = temp * mass* distance/ (distance + Eta)
                div +=  temp
        # div = div * density[i]
        div = div / density[i]
    
    return i, div

def pressure_possion(pos, vel, density, mass, kh, NN_idx, Eta, div_prev, p_type):
    n_part = len(density)
    max_iter = 1000
    div = div_prev
    pressure = np.zeros(n_part)
    iter = 0
    while iter < max_iter:
        args = [(i, pos, vel, density, mass, kh, NN_idx, Eta, p_type, div) for i in range(n_part)]
        div_vel_comp = div_part_vel(pos, vel, density, mass, kh, NN_idx, Eta, p_type, div, pressure, n_part)
        vel = vel - div_vel_comp
        div = calc_divergence(pos, vel, mass, kh, NN_idx, Eta, density, p_type)
        print(f'iter: {iter}, max_div: {np.max(np.abs(div))}')
        if iter%100 == 0:
            plot_prop(pos, div, f'div_{iter}')
        if np.max(np.abs(div)) < 1e-6:
            break
        iter += 1
    if iter == max_iter:
        print(f'Max iter reached and the max divergence is:{np.abs(np.max(div))}')
    return vel

def div_part_vel(pos, vel, density, mass, kh, NN_idx, Eta, p_type, div, pressure, n_part):
    h = kh
    h1 = 1/h
    A_mat = np.zeros((n_part, n_part))
    b_mat = np.zeros(n_part)
    weights_grad = ker_grad_arr(pos, kh, NN_idx)
    weights_lap = ker_lap_arr(pos, kh, NN_idx)
    args = [(i, pos, vel, density, mass, NN_idx, p_type, kh, weights_lap, weights_lap) for i in range(n_part)]
    pool = mp.Pool(6)
    results = pool.map(div_part_vel_part, args)

    # for i in range(n_part):
    #     temp_ii= 0.0
    #     for j in NN_idx[i]:
    #         if i ==j:
    #             continue
    #         r_ij = pos[i] - pos[j]
    #         distance = np.linalg.norm(r_ij)
    #         if distance < kh and distance > 0.0:
    #             # weight = gradient_poly6(r_ij, distance, kh)
    #             A_mat[i,j] =  (mass/density[i])*weights_lap[i,j]

    #             temp = np.dot((vel[j] - vel[i]), weights_grad[i,j])
    #             b_mat[i] += temp 
    #     b_mat[i] = b_mat[i] * mass / density[i]
    #     A_mat[i,i] = -np.sum(A_mat[i,:])
    for i, A_mat, b_mat in results:
        A_mat[i] = A_mat
        b_mat[i] = b_mat

    
    div_vel_comp = cg(A_mat, b_mat, atol = 1e-8)
    return div_vel_comp

def div_part_vel_part(arg):
    i, pos, vel, density, mass, NN_idx, p_type, kh, n_part, weights_lap, weights_grad = arg
    A_mat =np.zeros(n_part)
    b_mat = 0
    if p_type[i] == 0:
        return i, A_mat, b_mat

    temp_ii= 0.0
    for j in NN_idx[i]:
        if i ==j:
            continue
        r_ij = pos[i] - pos[j]
        distance = np.linalg.norm(r_ij)
        if distance < kh and distance > 0.0:
            # weight = gradient_poly6(r_ij, distance, kh)
            A_mat[j] =  (mass/density[i])*weights_lap[i,j]

            temp = np.dot((vel[j] - vel[i]), weights_grad[i,j])
            b_mat += temp 
    b_mat = b_mat * mass / density[i]
    A_mat[i] = -np.sum(A_mat[i,:])

    return i, A_mat, b_mat

def ker_grad_arr(pos, kh, NN_idx, vel):
    weights = np.zeros((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in NN_idx[i]:
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
            if distance < kh and distance > 0.0:
                temp = gradient_poly6(r_ij, distance, kh)
                temp = np.dot(temp, vel[NN_idx[j]])
                weights[i,j] = temp
    return weights
            
def ker_lap_arr(pos, kh, NN_idx):
    weights = np.zeros((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in NN_idx[i]:
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
            if distance < kh and distance > 0.0:
                temp = lap_poly6(r_ij, distance, kh)
                weights[i,j] = temp
    return weights

def lap_poly6(r_ij, distance, kh):
    h = kh
    h1 = 1/h
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    lap = 0.0
    if 0 < distance <= h:
        temp_2 = fac * (3 *(2 * temp *(4 *distance**2) + (temp * temp)*(-2)))
        temp_2 = temp_2 *  h1 / distance
        lap = temp_2
        return lap
    else:
        return lap





def main():
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    length = 1
    boundary_fac = 40
    dp = 0.004
    pos, vel, density, mass, p_type, kh, h, mid = make_particles(length, boundary_fac, dp)
    Eta = 1e-20
    radius_ = dp*15
    kh = radius_
    # radius_ = h*
    plot_velocity_vec(pos, vel, 'start')
    plot_prop(pos, p_type, 'p_type')
    nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)
    NN_idx = nbrs.radius_neighbors(pos)[1]
    print(f'average number of neighbors: {np.mean([len(idx) for idx in NN_idx])}')
    # plot_prop(pos, density, 'density')
    # density_new = calc_density(pos, mass, kh, NN_idx, Eta, density, p_type)
    # plot_prop(pos, density_new, 'density_new')
    # temp = (density-density_new)*100/density
    # plot_prop(pos, temp, 'per_cent_density_diff')
    div = calc_divergence(pos, vel, mass, kh, NN_idx, Eta, density, p_type)
    print(f'max_div: {np.max(np.abs(div))} and sum of div: {np.sum(div)}')
    plot_prop(pos, div, 'divergence_ini')
    vel = pressure_possion(pos, vel, density, mass, kh, NN_idx, Eta, div, p_type)
    plot_velocity_vec(pos, vel, 'end')

    


if __name__ == '__main__':
    main()
