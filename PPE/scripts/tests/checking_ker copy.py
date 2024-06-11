import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import numpy as np
import math

def make_particles(length =1, boundary_fac = 40, dp=0.008):
    # length =1 # meters
    # dp = 0.005 # meters
    # dp = 0.008 # meters
    # length =1 # meters
    # dp = 0.008 # meters
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
    temp = ((resolution*resolution/2 ))    
    # temp = temp + ((resolution )*2)
    temp = int(temp)
    return positions, velocity, density, mass, type, kh, h, temp

def poly6(kh, distance, no_part):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    if 0 < distance < h:
        no_part += 1
        return fac * (temp)**3
    else:
        return 0

def get_fac_w(h1):
    # get the kernel normalizing factor
    fac = 7/(4*np.pi)
    fac = fac * h1 * h1
    return fac

def ker_w(kh,distance, no_part):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    temp = 1 - 0.5 * q
    fac = get_fac_w(h1)
    if q <= 2 and q>0:
        no_part += 1
        return (fac * temp * temp * temp *temp *(2*q +1))
    else:
        return 0
def get_fac_cs(h1):
    # get the kernel normalizing factor
    fac = 10/(7 * np.pi)
    fac = fac * h1 * h1
    return fac

# Cubic spline kernel
def ker_cs(kh,distance, no_part):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = get_fac_cs(h1)
    tmp2 = 2. - q
    if (q > 2.0):
        no_part += 1
        val = 0.0
    elif (q > 1.0):
        no_part += 1
        val = 0.25 * tmp2 * tmp2 * tmp2
    else:
        no_part += 1
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac
def check_kernel_summation_for_particle(arg):
    i, positions, kh, NN_idx,mass, density= arg
    den = np.zeros(3)
    
    no_part = np.zeros(3)
    for j in NN_idx[0]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance <= kh and distance > 0.0:
            weight = poly6(kh, distance, no_part[0])
            den[0] += weight *mass
            weight = ker_w(kh, distance, no_part[1])
            den[1] += weight *mass
            weight = ker_cs(kh, distance, no_part[2])
            den[2] += weight *mass

            # den += weight *distance/(distance+1e-20)
    return no_part, den

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
    circle1 = plt.Circle((positions[idx,0], positions[idx,1]), kh_dsph, color='k', fill=False, linewidth=0.5)
    circle_poly = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[0]], color='r',linestyle = '--', fill=False, linewidth=0.5)
    circle_w = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[1]], color='b', linestyle = '--', fill=False, linewidth=0.5)
    circle_cs = plt.Circle((positions[idx,0], positions[idx,1]), h_vals[loc_tol[2]], color='g', linestyle = '--', fill=False, linewidth=0.5)
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
    
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle_poly)
    plt.gca().add_patch(circle_w)
    plt.gca().add_patch(circle_cs)
    # plt.gca().add_patch(circle2)

    plt.xlim(x1 - r_max, x2 + r_max)
    plt.ylim(y1 - r_max, y2 + r_max)

    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    print(positions[NN_idx[0][:],0].shape)
    plt.scatter(positions[NN_idx[0][:],0], positions[NN_idx[0][:],1], s=0.5, alpha = 0.5)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    # plt.show()
    plt.savefig(f'NN_grp_{title}.png')

def main():
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    length = 1
    boundary_fac = 40
    dp_list = np.linspace(0.02, 0.04, 4)
    # dp_list = np.linspace(0.002, 0.02, 10)
    dp_list = dp_list[::-1]
    dp_list = np.round(dp_list, 4)
    loc_tol = np.zeros((len(dp_list), 3))
    # dp_list = [0.008]
    no_part = np.zeros((len(dp_list), 3))
    for dp_i, dp in enumerate(dp_list):
        print(f'dp: {dp}')
        pos, vel, density, mass, type, kh, h, mid = make_particles(length, boundary_fac, dp)
        Eta = 1e-20
        max_kh_fac = length/2
        radius_ = max_kh_fac
        center_pt = np.array([0,0])
        nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)
        closest_to_center = nbrs.kneighbors([center_pt], 1)
        closest_index = closest_to_center[1].flatten()[0]
        closest_distance = closest_to_center[0].flatten()[0]
        print(f'closest_to_center particle: {closest_index} at distance: {closest_distance}')
        NN_idx_closet = nbrs.radius_neighbors(pos[closest_index].reshape(1,-1))[1]
        NN_idx_closet = NN_idx_closet.flatten()
        print(f'Number of neighbors of closest particle: {len(NN_idx_closet[0])}')
        h_vals = np.linspace(1.5*h, radius_, 1000)
        ker_vals = np.zeros((len(h_vals), 3))
        no_part_i = np.zeros((len(h_vals), 3))
        for i, h_val in enumerate(h_vals):
            kh_ = h_val
            arg =(closest_index, pos, kh_, NN_idx_closet, mass, density)
            no_part_i[i,:], ker_vals[i,:] = check_kernel_summation_for_particle(arg)
    
        # print(np.min(ker_vals[:,0]))
        # print(np.min(ker_vals[:,0]))

        try:
            temp = int(np.argwhere(ker_vals[:,0] > 200)[0][0])
            loc_tol[dp_i][0] = temp
            no_part[dp_i][0] = no_part_i[temp][0]
        except:
            loc_tol[dp_i][0] = 0
            no_part[dp_i][0] = 0
        try:
            temp = int(np.argwhere(ker_vals[:,0] > 200)[0][0])
            loc_tol[dp_i][1] = temp
            no_part[dp_i][1] = no_part_i[temp][1]
        except:
            loc_tol[dp_i][1] = 0
            no_part[dp_i][1] = 0
        try:
            temp = int(np.argwhere(ker_vals[:,0] > 200)[0][0])
            loc_tol[dp_i][2] = temp
            no_part[dp_i][2] = no_part_i[temp][2]
        except:
            loc_tol[dp_i][2] = 0
            no_part[dp_i][2] = 0

        loc_tol = loc_tol.astype(int)
        plot_NN_grp(pos, NN_idx_closet, f'{dp}', closest_index, h_vals, loc_tol[dp_i], ker_vals, dp, boundary_fac, kh)
        plt.cla()
        fac_dp_scale = dp
        plt.plot(h_vals/fac_dp_scale, (1000-ker_vals[:,0])/10, 'r', label = 'Poly6') 
        plt.plot(h_vals/fac_dp_scale, (1000-ker_vals[:,1])/10, 'b', label = 'Wendland' ) 
        plt.plot(h_vals/fac_dp_scale, (1000-ker_vals[:,2])/10, 'g', label = 'Cubic spline' ) 
        min = np.max(1000-ker_vals[:,0])/10
        plt.plot([kh/fac_dp_scale,kh/fac_dp_scale], [0,min],'k--', label='DSPH_default')
        plt.plot([h_vals[loc_tol[dp_i][0]]/fac_dp_scale,h_vals[loc_tol[dp_i][0]]/fac_dp_scale], [0,min],'r--')
        plt.plot([h_vals[loc_tol[dp_i][1]]/fac_dp_scale,h_vals[loc_tol[dp_i][1]]/fac_dp_scale], [0,min],'b--')
        plt.plot([h_vals[loc_tol[dp_i][2]]/fac_dp_scale,h_vals[loc_tol[dp_i][2]]/fac_dp_scale], [0,min],'g--')
        # plt.ylim(0,1)
        # plt.plot([2*h/fac_dp_scale,2*h/fac_dp_scale], [0,100],'k--', label='DSPH_default')
        plt.xlabel('kh (fac_dp_scaletor of dp)')
        plt.ylabel('Density, reference density = 1000')
        plt.grid(b='True', which = 'major', linestyle = '-')
        plt.grid(b='True', which = 'minor', linestyle = '--', alpha = 0.5)
        # plt.yscale('log')
        plt.gca().set_aspect('auto', adjustable='box')
        plt.legend(loc = 'upper right')
        plt.savefig(f'ker_vals_{dp}.png')
        plt.ylim(0,1)
        plt.savefig(f'ker_vals_zoom_{dp}.png')
    
    plt.cla()
    # plt.plot(dp_list, h_vals[loc_tol[:,0]]/dp_list[:], 'r', label = 'Poly6')
    # plt.plot(dp_list, h_vals[loc_tol[:,1]]/dp_list[:], 'b', label = 'Wendland')
    # plt.plot(dp_list, h_vals[loc_tol[:,2]]/dp_list[:], 'g', label = 'Cubic spline')
    plt.plot(dp_list, no_part[:,0], 'r', label = 'Poly6')
    plt.plot(dp_list, no_part[:,1], 'b', label = 'Wendland')
    plt.plot(dp_list, no_part[:,2], 'g', label = 'Cubic spline')
    plt.xlabel('kh (factor of dp)')
    plt.ylabel('Distance (factor of dp)')
    plt.xticks(dp_list)
    plt.ticklabel_format(axis='x', scilimits =[-1,1])
    plt.savefig('loc_tol.png', dpi=600)

if __name__ == "__main__":
    main()