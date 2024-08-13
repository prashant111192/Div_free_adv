import multiprocessing as mp
from multiprocessing import sharedctypes as sct
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import scipy.sparse as sp
import numpy as np
import math
import time
import sys

def make_particles(length =1, boundary_fac = 40, dp=0.008):
    boundary_width = dp*boundary_fac # meters

    x1 = -length/2 - boundary_width/2
    x2 = length/2 + boundary_width/2
    y1 = -length/2 - boundary_width/2
    y2 = length/2 + boundary_width/2

    resolution = int((length+boundary_width)/dp)
    pos = np.zeros((resolution**2,2), dtype=np.float32)
    density = np.ones((resolution**2), dtype=np.float32)
    density = density * 1000
    velocity = np.zeros((resolution**2,2), dtype=np.float32)
    mass = density[1]*dp**2
    p_type = np.zeros((resolution**2), dtype=np.float32)

    count = 0
    for i in range(resolution):
        for j in range(resolution):
            pos[count] = np.array([(x1+(dp*i)),(y1+(dp*j))], dtype=np.float32)
            if x1+boundary_width/2<(x1+(dp*i))<x2-dp-boundary_width/2 and y1+boundary_width/2<(y1+(dp*j))<y2-dp-boundary_width/2:
                p_type[count] = 1 # Fluid particle
                # velocity[count] = np.array([math.sin(pos[count][0])*0.01,math.cos(pos[count][1])*0.01])
                if pos[count][0] > 0 and pos[count][1] > 0 and pos[count][0] < 30*dp and pos[count][1] < 30*dp:
                    velocity[count] = np.array([0.05,0.05], dtype=np.float32)
            count += 1
    
    # h = 0.008660
    h_const = 0.02 
    dp_const = 0.008
    h_fac = h_const/dp_const
    h = h_fac * dp

    kh = h*2
    mid = ((resolution*resolution/2 ))    
    mid = int(mid)
    print(f'Number of particles: {len(pos)}')
    return pos, velocity, density, mass, p_type, kh, h, mid

def make_particles_1(length =1, boundary_fac = 40, dp=0.008):
    boundary_width = dp*boundary_fac # meters

    x1 = -length/2 - boundary_width/2
    x2 = length/2 + boundary_width/2
    y1 = -length/2 - boundary_width/2
    y2 = length/2 + boundary_width/2

    resolution = int((length+boundary_width)/dp)
    pos = np.zeros((resolution**2,2), dtype=np.float32)
    density = np.ones((resolution**2), dtype=np.float32)
    density = density * 1000
    velocity = np.zeros((resolution**2,2), dtype=np.float32)
    mass = density[1]*dp**2
    p_type = np.zeros((resolution**2), dtype=np.float32)

    count = 0
    for i in range(resolution):
        for j in range(resolution):
            pos[count] = np.array([(x1+(dp*i)),(y1+(dp*j))], dtype=np.float32)
            if x1+boundary_width/2<(x1+(dp*i))<x2-dp-boundary_width/2 and y1+boundary_width/2<(y1+(dp*j))<y2-dp-boundary_width/2:
                p_type[count] = 1 # Fluid particle
                # velocity[count] = np.array([math.sin(pos[count][0])*0.01,math.cos(pos[count][1])*0.01])
                if pos[count][0] > 0 and pos[count][1] > 0 and pos[count][0] < length * 0.15 and pos[count][1] < length * 0.15:
                    velocity[count] = np.array([0.05,0.05], dtype=np.float32)
            count += 1
    
    # h = 0.008660
    h_const = 0.02 
    dp_const = 0.008
    h_fac = h_const/dp_const
    h = h_fac * dp

    kh = h*2
    mid = ((resolution*resolution/2 ))    
    mid = int(mid)
    print(f'Number of particles: {len(pos)}')
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
    # pos = positions[::25,:]
    # vel = velocity[::25,:]
    pos = positions
    vel = velocity
    plt.cla()
    plt.quiver(pos[:,0], pos[:,1], vel[:,0], vel[:,1], scale=1, scale_units='xy')
    # plt.quiver(positions[:,0], positions[:,1], velocity[:,0], velocity[:,1], scale=0.3, scale_units='xy')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'velocity_vec_{title}.png')

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='jet', s=10, alpha=1)
    # plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='twilight_shifted', s=10, alpha=1)
    # plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='viridis', s=0.2, alpha=0.5)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{title}.png')

def calc_density(positions, mass, kh, NN_idx, Eta, den_orig, type):
    density = np.zeros(len(positions), dtype=np.float32)
    results = np.zeros(len(positions), dtype=np.float32)
    res_idx = np.zeros(len(positions), dtype=int)
    pool = mp.Pool((mp.cpu_count()-6))
    # pool = mp.Pool(6)
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
    if type[i] == 0: # boundary particle
        return i, density[i]
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
    divergence = np.zeros(len(density), dtype=np.float32)
    results = np.zeros(len(density), dtype=np.float32)
    pool = mp.Pool((mp.cpu_count()-6))
    args = [(i, pos, vel , density, mass, kh, NN_idx, Eta, p_type) for i in range(n_particles)]
    # for arg in args:
    #     i, div = calc_divergence_part(arg)
    #     divergence[i] = div
    results = pool.map(calc_divergence_part, args)
    pool.close()
    pool.join()
    for i, div in results:
        divergence[i] = div
    
    print(f'total abs sum of divergence: {np.sum(np.abs(divergence))}')
    return divergence

def gradient_poly6(r_ij, distance, kh):
    h = kh
    h1 = 1/h
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    grad = np.zeros(2, dtype=np.float32)
    if 0 < distance <= h:
        # temp_2 = fac * 3*((temp)**2)*(-2*distance) / distance
        temp_2 = fac * (-6)*((temp)**2)
        grad = temp_2 * r_ij

        return grad
    else:
        return grad 


def calc_divergence_part(arg):
    i, pos, vel, density, mass, kh, NN_idx, Eta, p_type = arg

    div = 0.0
    if p_type[i] == 0:
        return i, div
        
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

def pressure_possion(pos, vel, density, mass, kh, NN_idx, Eta, div_prev, p_type, ker_lap_array, ker_grad_array_x, ker_grad_array_y):
    n_part = len(density)
    max_iter = 1000
    div = div_prev
    pressure= 0
    iter = 0

    while iter < max_iter:
        div_vel_comp = div_part_vel(pos, vel, density, mass, kh, NN_idx, Eta, p_type, div, pressure, n_part, ker_lap_array, ker_grad_array_x, ker_grad_array_y)
        # vel = vel - div_vel_comp
        q = calc_non_div_vel(pos, div_vel_comp, density, mass, kh, NN_idx, ker_grad_array_x, ker_grad_array_y, p_type)
        vel = vel - q
        div = calc_divergence(pos, vel, mass, kh, NN_idx, Eta, density, p_type)
        print(f'iter:{iter}:max_div:{np.max(np.abs(div))}:div_sum:{np.sum(div)}')
        if iter%5 == 0:
            plot_prop(pos, div, f'div_{iter}', 0.4, -0.4)
            plot_velocity_vec(pos, vel, f'vel_{iter}')
        if np.max(np.abs(div)) < 1e-6:
            break
        iter += 1
    if iter == max_iter:
        print(f'Max iter reached and the max divergence is:{np.abs(np.max(div))}')
    return vel

def calc_non_div_vel(pos, q, density, mass, kh, NN_idx, ker_grad_array_x, ker_grad_array_y, p_type):
    n_part = len(density)
    grad_q = np.zeros((n_part, 2), dtype=np.float32)
    for i in range(n_part):
        if p_type[i] == 0:
            for j in NN_idx[i]:
                r_ij = pos[i] - pos[j]
                distance = np.linalg.norm(r_ij)
                if distance < kh and distance > 0.0:
                    temp = np.array([ker_grad_array_x[i,j], ker_grad_array_y[i,j]])
                    # temp = gradient_poly6(r_ij, distance, kh)
                    temp = temp *(q[j] - q[i])
                    grad_q[i] += temp
        else:
            for j in NN_idx[i]:
                r_ij = pos[i] - pos[j]
                distance = np.linalg.norm(r_ij)
                if distance < kh and distance > 0.0:
                    temp = np.array([ker_grad_array_x[i,j], ker_grad_array_y[i,j]])
                    # temp = gradient_poly6(r_ij, distance, kh)
                    temp = temp *(q[j] - q[i])
                    grad_q[i] += temp
        grad_q[i] = grad_q[i] * mass / density[i]
    # q = q - grad_q
    return grad_q

def div_part_vel(pos, vel, density, mass, kh, NN_idx, Eta, p_type, div, pressure, n_part, weights_lap, weights_grad_x, weights_grad_y):
    # h = kh
    # h1 = 1/h
    A_mat = sp.lil_matrix((n_part, n_part), dtype=np.float32)
    b_mat = np.zeros(n_part, dtype=np.float32)
    # start = time.time()
    # b_mat = sp.lil_matrix(1, n_part)
    # end = time.time()
    # A_mat = np.zeros((n_part, n_part))
    # b_mat = np.zeros(n_part)
    # start = time.time()
    # weights_grad = ker_grad_arr(pos, kh, NN_idx, vel)
    # weights_lap = ker_lap_arr(pos, kh, NN_idx)
    # end = time.time()
    args = [(i, pos, vel, density, mass, NN_idx, p_type, kh, n_part, weights_lap, weights_grad_x, weights_grad_y) for i in range(n_part)]
    pool = mp.Pool((mp.cpu_count()-6))
    # pool = mp.Pool(6)
    # start = time.time()
    results = pool.map(div_part_vel_part, args)
    # end = time.time()
    # print(f'Time taken to calculate A and b: {end-start} s')

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
    for i, A_mat_, b_mat_ in results:
        A_mat[i,:] = A_mat_
        b_mat[i] = b_mat_

    
    np.savetxt("b_mat.csv", b_mat, delimiter=",")
    print(f' Number of stored values in A: {A_mat.count_nonzero}')
    temmpp = A_mat.multiply(A_mat).sum()
    print(f'test_A: {temmpp}')
    temmpp = (b_mat*b_mat).sum()
    print(f'test_b: {temmpp}')
    # exit(0)
    x = cg(A_mat, b_mat, atol = 1e-5)
    div_vel_comp, cg_success = np.array(x[0], dtype=np.float32), x[1]
    if cg_success != 0:
        print('CG did not converge')
    # div_vel_comp = np.array(cg(A_mat, b_mat, atol = 1e-8)[])
    # print(div_vel_comp.shape)
    return div_vel_comp

def div_part_vel_part(arg):
    i, pos, vel, density, mass, NN_idx, p_type, kh, n_part, weights_lap, weights_grad_x, weights_grad_y = arg
    # A_mat = sp.lil_matrix((n_part, 1))
    A_mat =np.zeros(n_part, dtype=np.float32)
    b_mat = 0
    if p_type[i] == 0:
        A_mat = sp.lil_matrix(A_mat, dtype=np.float32)
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

            temp = np.dot((vel[j] - vel[i]), np.array((weights_grad_x[i,j], weights_grad_y[i,j]), dtype=np.float32))
            b_mat += temp 
    b_mat = b_mat * mass / density[i]
    A_mat[i] = -np.sum(A_mat[:])

    A_mat = sp.lil_matrix(A_mat, dtype=np.float32)
    

    return i, A_mat, b_mat

# def ker_grad_arr(pos, kh, NN_idx):
#     weights = sp.lil_matrix((len(pos), len(pos)))
#     # weights = np.zeros((len(pos), len(pos)))
#     for i in range(len(pos)):
#         for j in NN_idx[i]:
#             r_ij = pos[i] - pos[j]
#             distance = np.linalg.norm(r_ij)
#             if distance < kh and distance > 0.0:
#                 temp = gradient_poly6(r_ij, distance, kh)
#                 temp = np.dot(temp, vel[j])
#                 weights[i,j] = temp
#     return weights
            
# def ker_lap_arr(pos, kh, NN_idx):
#     weights = sp.lil_matrix((len(pos), len(pos)))
#     # weights = np.zeros((len(pos), len(pos)))
#     for i in range(len(pos)):
#         for j in NN_idx[i]:
#             r_ij = pos[i] - pos[j]
#             distance = np.linalg.norm(r_ij)
#             if distance < kh and distance > 0.0:
#                 temp = lap_poly6(r_ij, distance, kh)
#                 weights[i,j] = temp
#     return weights

def lap_poly6(r_ij, distance, kh):
    h = kh
    h1 = 1/h
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    lap = 0.0
    if 0 < distance and distance <= h:
        temp_2 = fac * (3 *(2 * temp *(4 *distance**2) + (temp * temp)*(-2)))
        # temp_2 = temp_2 
        lap = temp_2
        return lap
    else:
        return lap


def abs_sum(sp_mat):
    for i in range(len(sp_mat)):
        for j in range(len(sp_mat[i])):
            sp_mat[i,j] = abs(sp_mat[i,j])
    return sp_mat


def main():
    start = time.time()
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    length = 390
    boundary_fac = 10
    dp = 1
    # dp = 0.006
    pos, vel, density, mass, p_type, kh, h, mid = make_particles(length, boundary_fac, dp)
    Eta = 1e-20
    radius_ = dp*3
    kh = radius_
    end = time.time()
    print(f'Time taken to make particles: {end-start} s')
    # radius_ = h*
    plot_velocity_vec(pos, vel, 'start')
    plot_prop(pos, p_type, 'p_type')

    nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)
    NN_idx = nbrs.radius_neighbors(pos)[1]
    # new_NN_idx = []
    # for i in range(len(NN_idx)):
    #     temp = []
    #     for j in range(len(NN_idx[i])):
    #         if i != j:
    #             temp.append(NN_idx[i][j])
    #     new_NN_idx.append(temp)
    # NN_idx = new_NN_idx
    print(f'max number of neighbors: {np.max([len(idx) for idx in NN_idx])}')
    print(f'total number of neighbors: {np.sum([len(idx) for idx in NN_idx])}')
    print(f'average number of neighbors: {np.mean([len(idx) for idx in NN_idx])}')

    start = time.time()

    print('started calculating kernels')
    ker_grad_array_x, ker_grad_array_y = ker_grad_arr(pos, kh, NN_idx)
    ker_lap_array = ker_lap_arr(pos, kh, NN_idx)
    end = time.time()
    
    print(f'abs_sum_grad_x: {ker_grad_array_x.multiply(ker_grad_array_x).sum()}')
    print(f'Time taken to calculate kernels: {end-start} s')
    # test__ = (ker_lap_array.multiply(ker_lap_array)).sum()
    # print(f'test_lap: {test__}')
    print(ker_grad_array_x.sum())
    print(F'TRACE of ker_grad_array_x: {ker_grad_array_x.sum()}')
    # print(f' abs_sum ={abs_sum(ker_grad_array_x).sum()}')
    # print(f'max_grad_x: {ker_grad_array_x.max()}')
    # print('lap')
    # print(f'Number of non0 in gradx: {ker_grad_array_x.count_nonzero}, in grady: {ker_grad_array_y.count_nonzero}, in lap: {ker_lap_array.count_nonzero}, in lap: {ker_lap_array.count_nonzero}')
    # plot_prop(pos, density, 'density')
    # density_new = calc_density(pos, mass, kh, NN_idx, Eta, density, p_type)
    # plot_prop(pos, density_new, 'density_new')
    # temp = (density-density_new)*100/density
    # plot_prop(pos, temp, 'per_cent_density_diff')
    div = calc_divergence(pos, vel, mass, kh, NN_idx, Eta, density, p_type)
    print(f'max_div:{np.max(np.abs(div))}:div_sum:{np.sum(div)}')

    plot_prop(pos, div, 'divergence_ini')
    vel = pressure_possion(pos, vel, density, mass, kh, NN_idx, Eta, div, p_type, ker_lap_array, ker_grad_array_x, ker_grad_array_y)
    plot_velocity_vec(pos, vel, 'end')

def get_sparse_size(matrix):
    data_size = sum(len(row) * sys.getsizeof(row[0]) for row in matrix.data if row)
    rows_size = sum(len(row) * sys.getsizeof(row[0]) for row in matrix.rows if row)
    return int((data_size + rows_size) / 1024.)

def ker_grad_arr(pos, kh, NN_idx):
    # weights = np.zeros((len(pos), len(pos), 2))
    weights_x = sp.lil_matrix((len(pos), len(pos)), dtype=np.float32)
    weights_y = sp.lil_matrix((len(pos), len(pos)), dtype=np.float32)
    args = [(i, pos, kh, NN_idx) for i in range(len(pos))]
    pool = mp.Pool((mp.cpu_count()-6))
    results = pool.map(ker_grad_arr_part, args)
    pool.close()
    pool.join()
    tt = 0
    for i, weights in results:
        for ji, j in enumerate(NN_idx[i]):
            tt = tt+weights_x[i,j]
            weights_x[i, j] = weights[ji, 0]
            weights_y[i, j] = weights[ji, 1]
    print(f'done')
    test = (abs(weights_x)).sum()
    # test = abs(weights_x).sum
    # test = (weights_x.multiply(weights_x)).sum()
    print(f'sum of grad_x: {tt}')
    print(f'test_x: {test}')
    test = (weights_y.multiply(weights_y).divide(abs(weights_y))).sum()
    print(f'test_y: {test}')
    # for i in range(len(pos)):
    #     for j in NN_idx[i]:
    #         r_ij = pos[i] - pos[j]
    #         distance = np.linalg.norm(r_ij)
    #         if distance < kh and distance > 0.0:
    #             temp = gradient_poly6(r_ij, distance, kh)
    #             weights_x[i,j] = temp[0]
    #             weights_y[i,j] = temp[1]
    #             # weights[i,j,:] = temp
    return weights_x, weights_y

def ker_grad_arr_part(arg):
    i, pos, kh, NN_idx = arg
    weights = sp.lil_matrix((len(NN_idx[i]),2), dtype=np.float32)
    # weights = np.zeros((len(NN_idx[i]),2), dtype=np.float32)

    for ij, j in enumerate(NN_idx[i]):
        if i == j:
            continue
        r_ij = pos[i] - pos[j]
        distance = np.linalg.norm(r_ij)
        if distance <= kh and distance > 0.0:
            weights[ij] = gradient_poly6(r_ij, distance, kh)
            # weights[i,j,:] = temp
    
    return i, weights

def ker_lap_arr(pos, kh, NN_idx):
    weights = sp.lil_matrix((len(pos), len(pos)), dtype=np.float32)
    # weights = np.zeros((len(pos), len(pos)), dtype=np.float32)
    for i in range(len(pos)):
        for j in NN_idx[i]:
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
            if distance <= kh and distance > 0.0:
                temp = lap_poly6(r_ij, distance, kh)
                weights[i,j] = temp
    return weights
    
if __name__ == '__main__':
    main()
