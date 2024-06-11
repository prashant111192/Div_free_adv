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
    
    # h = 0.008660
    h_const = 0.02 
    dp_const = 0.008
    h_fac = h_const/dp_const
    h = h_fac * dp

    kh = h*2
    mid = ((resolution*resolution/2 ))    
    mid = int(mid)
    return positions, velocity, density, mass, type, kh, h, mid

def poly6(kh, distance, count_poly6):
    h = kh
    # h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    if 0 < distance <= h:
        count_poly6 += 1
        return fac * (temp)**3, count_poly6
    else:
        return 0, count_poly6

def get_fac_w(h1):
    # get the kernel normalizing factor
    fac = 7/(4*np.pi)
    fac = fac * h1 * h1
    return fac
def ker_w(kh,distance, count_w):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    temp = 1 - 0.5 * q
    fac = get_fac_w(h1)
    if q <= 2 and q>0:
        count_w += 1
        return (fac * temp * temp * temp *temp *(2*q +1)), count_w

    else:
        return 0, count_w

def get_fac_cs(h1):
    # get the kernel normalizing factor
    fac = 10/(7 * np.pi)
    fac = fac * h1 * h1
    return fac
# Cubic spline kernel
def ker_cs(kh,distance, count_cs):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = get_fac_cs(h1)
    tmp2 = 2. - q
    if (q >= 2.0):
        val = 0.0
    elif (q > 1.0):
        count_cs += 1
        val = 0.25 * tmp2 * tmp2 * tmp2
    else:
        count_cs += 1
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac, count_cs

def check_kernel_summation_for_particle(arg):
    i, positions, kh, NN_idx,mass, density= arg
    den = np.zeros(3)
    count = np.zeros(3)
    for j in NN_idx:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance <= kh and distance > 0.0:
            # weight, count = [poly6(kh, distance, count[0]), ker_w(kh, distance, count[1]), ker_cs(kh, distance, count[2])]
            # den += np.array(weight) * mass
            weight, count[0] = poly6(kh, distance, count[0])
            den[0] += weight *mass
            weight, count[1] = ker_w(kh, distance, count[1])
            den[1] += weight *mass
            weight, count[2] = ker_cs(kh, distance, count[2])
            den[2] += weight *mass

            # den += weight *distance/(distance+1e-20)
    return count, den

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
        print(den)
    return summatiion

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:,0], positions[:,1], c=prop, cmap='viridis', s=2)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.savefig(f'{title}.png')

def plot_NN_grp(positions, NN_idx, title, mid, h_vals, loc_tol, ker_vals, dp, boundary_fac, kh_dsph):
    plt.cla()
    x1 = np.min(positions[:,0])
    x2 = np.max(positions[:,0])
    y1 = np.min(positions[:,1])
    y2 = np.max(positions[:,1])
    r_min = np.min(h_vals)
    r_max = np.max(h_vals)
        # Plotting the square
    # for k in NN_idx[0]:
    #     distance = np.linalg.norm(positions[k] - positions[mid])
    #     if distance == 0:
    #         idx = k
    #         break
    idx = mid
    circle1 = plt.Circle((positions[idx,0], positions[idx,1]), kh_dsph, color='k', fill=False, linewidth=0.5, label = 'DSPH_default')
    plt.gca().add_patch(circle1)
    if loc_tol[0] != 0:
        circle_poly = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[0]], color='r',linestyle = '--', fill=False, linewidth=0.5, label = 'Poly6')
        # circle_poly = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[0]]/2, color='r',linestyle = '--', fill=False, linewidth=0.5, label = 'Poly6')
        plt.gca().add_patch(circle_poly)
    if loc_tol[1] != 0:
        circle_w = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[1]], color='b', linestyle = '--', fill=False, linewidth=0.5, label = 'Wendland')
        plt.gca().add_patch(circle_w)
    if loc_tol[2] != 0:
        circle_cs = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[2]], color='g', linestyle = '--', fill=False, linewidth=0.5, label = 'Cubic spline')
        plt.gca().add_patch(circle_cs)
    # circle2 = plt.Circle((pos[idx,0], pos[idx,1]), r_max, color='r', fill=False)
    # plt.figure(figsize=(8, 8))
    plt.plot([x1, x2], [y1, y1], 'k-')  # bottom
    plt.plot([x1, x2], [y2, y2], 'k-')  # top
    plt.plot([x1, x1], [y1, y2], 'k-')  # left
    plt.plot([x2, x2], [y1, y2], 'k-')  # right
    x1 = x1 + (dp*boundary_fac/2)
    x2 = x2 - (dp*boundary_fac/2)
    y1 = y1 + (dp*boundary_fac/2)
    y2 = y2 - (dp*boundary_fac/2)
    plt.plot([x1, x2], [y1, y1], 'b--')  # bottom
    plt.plot([x1, x2], [y2, y2], 'b--')  # top
    plt.plot([x1, x1], [y1, y2], 'b--')  # left
    plt.plot([x2, x2], [y1, y2], 'b--')  # right

    # Plotting the circles
    
    # plt.gca().add_patch(circle2)

    plt.xlim(x1 - r_max, x2 + r_max)
    plt.ylim(y1 - r_max, y2 + r_max)

    plt.title(f'Kernel size for dp = {title}m')
    plt.gca().set_aspect('equal', adjustable='box')

    # print(positions[NN_idx[:],0].shape)
    plt.scatter(positions[NN_idx[:],0], positions[NN_idx[:],1], s=0.5, alpha = 0.5)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    # plt.show()
    plt.legend(loc='upper right')
    plt.savefig(f'NN_grp_{title}.png')

def main():
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    length = 1
    boundary_fac = 40

    dp_list = np.linspace(0.001, 0.01, 10)
    # dp_list = np.linspace(0.002, 0.02, 10)
    dp_list = dp_list[::-1]
    dp_list = np.round(dp_list, 4)
    # dp_list = [0.008]

    tolerance = 0.01
    loc_tol = np.zeros((len(dp_list), 3))

    count_tol = np.zeros((len(dp_list), 3))
    for dp_i, dp in enumerate(dp_list):
        print(f'dp: {dp}')

        center_pt = np.array([0,0])

        pos, vel, density, mass, type, kh, h, mid = make_particles(length, boundary_fac, dp)
        Eta = 1e-20

        max_kh_fac = length/2
        radius_ = dp*40 # for the nn group
        # radius_ = max_kh_fac # for the nn group
        nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)

        closest_to_0 = nbrs.kneighbors([center_pt], 1)
        distance_0 = closest_to_0[0].flatten()[0]
        idx_0 = closest_to_0[1].flatten()[0]


        print(f'closest_to_0 particle: {idx_0} at distance: {distance_0}')
        NN_idx_closet = nbrs.radius_neighbors(pos[idx_0].reshape(1,-1))[1]
        NN_idx_closet = NN_idx_closet.flatten()[0]
        print(f'Number of neighbors of closest particle: {len(NN_idx_closet)}')

        h_vals = np.linspace(1.5*h, radius_, 200) # starts from 1.5*h. *****List of h not kh*****
        ker_vals = np.zeros((len(h_vals), 3)) # results from the kernel summation

        count = np.zeros((len(h_vals), 3))
        for i, h_val in enumerate(h_vals):
            kh_ = h_val
            arg =(idx_0, pos, kh_, NN_idx_closet, mass, density)
            count[i,:], ker_vals[i,:] = check_kernel_summation_for_particle(arg)
    
        print(np.min(ker_vals[:,0]))

        for i in range(3):
            try:
                loc_tol[dp_i,i] = ((np.argwhere(ker_vals[:,i] > 1000*(1-tolerance))[0][0]))
                count_tol[dp_i,i] = count[loc_tol[dp_i,i],i]
            except:
                loc_tol[dp_i,i] = 0
        loc_tol = loc_tol.astype(int)

        plot_NN_grp(pos, NN_idx_closet, f'{dp}', idx_0, h_vals, loc_tol[dp_i], ker_vals, dp, boundary_fac, kh)

        plt.cla()
        plt.clf()
        fac_dp_scale = dp
        # fac_dp_scale = dp
        ker_per_cent = (1000-ker_vals)/10
        plt.plot(h_vals/(fac_dp_scale), ker_per_cent[:,0], 'r', label = 'Poly6') 
        # plt.plot(h_vals/(fac_dp_scale*2), ker_per_cent[:,0], 'r', label = 'Poly6') 
        plt.plot(h_vals/fac_dp_scale, ker_per_cent[:,1], 'b', label = 'Wendland' ) 
        plt.plot(h_vals/fac_dp_scale, ker_per_cent[:,2], 'g', label = 'Cubic spline' ) 
        min = np.max(1000-ker_vals[:,0])/10
        plt.plot([kh/fac_dp_scale,kh/fac_dp_scale], [0,min],'k--', label='DSPH_default')
        if loc_tol[dp_i][0] != 0:
            plt.plot([h_vals[loc_tol[dp_i][0]]/(fac_dp_scale),h_vals[loc_tol[dp_i][0]]/(fac_dp_scale)], [0,min],'r--')
            # plt.plot([h_vals[loc_tol[dp_i][0]]/(fac_dp_scale*2),h_vals[loc_tol[dp_i][0]]/(fac_dp_scale*2)], [0,min],'r--')
        if loc_tol[dp_i][1] != 0:
            plt.plot([h_vals[loc_tol[dp_i][1]]/fac_dp_scale,h_vals[loc_tol[dp_i][1]]/fac_dp_scale], [0,min],'b--')
        if loc_tol[dp_i][2] != 0:
            plt.plot([h_vals[loc_tol[dp_i][2]]/fac_dp_scale,h_vals[loc_tol[dp_i][2]]/fac_dp_scale], [0,min],'g--')
        # plt.ylim(0,1)
        # plt.plot([2*h/fac_dp_scale,2*h/fac_dp_scale], [0,100],'k--', label='DSPH_default')
        plt.xlabel('kh (fac_dp_scaletor of dp)')
        plt.ylabel('% difference in reference, value')
        plt.grid(b='True', which = 'major', linestyle = '-')
        plt.grid(b='True', which = 'minor', linestyle = '--', alpha = 0.5)
        # plt.yscale('log')
        plt.gca().set_aspect('auto', adjustable='box')
        plt.legend(loc = 'upper right')
        plt.title(f'Kernel values dp = {dp}m')
        plt.savefig(f'ker_vals_{dp}.png')
        plt.ylim(0,1)
        plt.title(f'Kernel values dp = {dp}m, zoomed')
        plt.savefig(f'ker_vals_zoom_{dp}.png')
    
    plt.cla()
    plt.plot(dp_list, h_vals[loc_tol[:,0]]/dp_list[:], 'r', label = 'Poly6')
    plt.plot(dp_list, h_vals[loc_tol[:,1]]/dp_list[:], 'b', label = 'Wendland')
    plt.plot(dp_list, h_vals[loc_tol[:,2]]/dp_list[:], 'g', label = 'Cubic spline')
    plt.xlabel('kh (factor of dp)')
    plt.ylabel(f'Distance (*dp)')
    plt.xticks(dp_list)
    plt.grid(b='True', which = 'major', linestyle = '-')
    plt.ticklabel_format(axis='x', scilimits =[-1,1])
    plt.legend(loc='upper right')
    plt.title(f'Distane v/s dp for {tolerance*100}% error')
    plt.savefig('loc_tol.png', dpi=600)

    plt.cla()
    plt.plot(dp_list, count_tol[:,0], 'r', label = 'Poly6')
    plt.plot(dp_list, count_tol[:,1], 'b', label = 'Wendland')
    plt.plot(dp_list, count_tol[:,2], 'g', label = 'Cubic spline')
    plt.xlabel('dp (m)')
    plt.ylabel(f'Number of particles')
    plt.xticks(dp_list)
    plt.ticklabel_format(axis='x', scilimits =[-1,1])
    plt.grid(b='True', which = 'major', linestyle = '-')
    plt.legend(loc='lower left')
    plt.title(f'Number of particles for {tolerance*100}% error')
    plt.savefig('count_tol.png', dpi=600)

if __name__ == "__main__":
    main()