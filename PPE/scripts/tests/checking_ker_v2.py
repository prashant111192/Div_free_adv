
import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.neighbors as NN
from scipy.sparse.linalg import cg
import numpy as np
import math

# class particle:
#     pos = np.array([0,0])
#     vel = np.array([0,0])
#     density = 1000
#     p_type = 0 

    # def __init__(self, pos, vel, density):
    #     self.pos = pos
    #     self.vel = vel
    #     self.density = density

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
    p_type = np.zeros(resolution**2)

    count = 0
    for i in range(resolution):
        for j in range(resolution):
            positions[count] = np.array([(x1+(dp*i)),(y1+(dp*j))])
            if x1+boundary_width/2<(x1+(dp*i))<x2-dp-boundary_width/2 and y1+boundary_width/2<(y1+(dp*j))<y2-dp-boundary_width/2:
                p_type[count] = 1
            count += 1
    
    # h = 0.008660
    h_const = 0.02 
    dp_const = 0.008
    h_fac = h_const/dp_const
    h = h_fac * dp

    kh = h*2
    mid = ((resolution*resolution/2 ))    
    mid = int(mid)
    return positions, velocity, density, mass, p_type, kh, h, mid

def poly6(kh, distance, count_poly6):
    h = kh/2
    h1 = 1/h
    q = distance * h1
    fac = (4*h1**8)/(np.pi)
    temp = h**2 - distance**2
    if 0 < distance < h:
        count_poly6 += 1
        return fac * (temp)**3, count_poly6
    else:
        return 0, count_poly6

def get_fac_w(h1):
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
    if (q > 2.0):
        val = 0.0
    elif (q > 1.0):
        count_cs += 1
        val = 0.25 * tmp2 * tmp2 * tmp2
    else:
        count_cs += 1
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac, count_cs

def main():
    length = 1
    boundary_fac = 40

    dp_list = np.linspace(0.008, 0.02, 4)
    # dp_list = np.linspace(0.002, 0.02, 10)
    dp_list = dp_list[::-1]
    dp_list = np.round(dp_list, 4)
    # dp_list = [0.008]

    loc_tol = np.zeros((len(dp_list), 3))
    count_tol = np.zeros((len(dp_list), 3))

    for dp_i, dp in enumerate(dp_list):
        print(f'dp: {dp}')

        center_pt = np.array([0,0])

        pos, vel, density, mass, type, kh, h, mid = make_particles(length, boundary_fac, dp)
        Eta = 1e-20

        max_kh_fac = length/2 - dp
        radius_ = max_kh_fac # for the nn group
        nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)

        closest_to_0 = nbrs.kneighbors([center_pt], 1)
        idx_0 = closest_to_0[1].flatten()[0]
        distance_0 = closest_to_0[0].flatten()[0]

        print(f'closest_to_0 particle: {idx_0} at distance: {distance_0}')
        NN_idx_closet = nbrs.radius_neighbors(pos[idx_0].reshape(1,-1))[1]
        NN_idx_closet = NN_idx_closet.flatten()[0]
        print(f'Number of neighbors of closest particle: {len(NN_idx_closet)}')

        h_vals = np.linspace(1.5*h, radius_, 100) # starts from 1.5*h. *****List of h not kh*****
        ker_vals = np.zeros((len(h_vals), 3)) # results from the kernel summation

        count = np.zeros((len(h_vals), 3))
        for i, h_val in enumerate(h_vals):
            kh_ = h_val
            arg =(idx_0, pos, kh_, NN_idx_closet, mass, density)
            count[i,:], ker_vals[i,:] = check_kernel_summation_for_particle(arg)
    
        print(np.min(ker_vals[:,0]))

        for i in range(3):
            try:
                temp = ((np.argwhere(ker_vals[:,i] > 990)[0][0]))
                loc_tol[dp_i,i] = h_vals[temp]
                count_tol[dp_i,i] = count[loc_tol[dp_i,i],i]
            except:
                loc_tol[dp_i,i] = 0
        # try:
        #     loc_tol[dp_i][0] = int((np.argwhere(ker_vals[:,0] > 900)[0][0]))
        # except:
        #     loc_tol[dp_i][0] = 0
        # try:
        #     loc_tol[dp_i][1] = int((np.argwhere(ker_vals[:,1] > 900)[0][0]))
        # except:
        #     loc_tol[dp_i][1] = 0
        # try:
        #     loc_tol[dp_i][2] = int((np.argwhere(ker_vals[:,2] > 900)[0][0]))
        # except:
        #     loc_tol[dp_i][2] = 0
        loc_tol = loc_tol.astype(int)

        plot_NN_grp(pos, NN_idx_closet, f'{dp}', idx_0, h_vals, loc_tol[dp_i], ker_vals, dp, boundary_fac, kh)

        plt.cla()
        plt.clf()
        fac_dp_scale = dp
        ker_per_cent = (1000-ker_vals)/10
        plt.plot(h_vals/(fac_dp_scale*2), ker_per_cent[:,0], 'r', label = 'Poly6') 
        plt.plot(h_vals/fac_dp_scale, ker_per_cent[:,1], 'b', label = 'Wendland' ) 
        plt.plot(h_vals/fac_dp_scale, ker_per_cent[:,2], 'g', label = 'Cubic spline' ) 
        min = np.max(1000-ker_vals[:,0])/10
        plt.plot([kh/fac_dp_scale,kh/fac_dp_scale], [0,min],'k--', label='DSPH_default')
        plt.plot([h_vals[loc_tol[dp_i][0]]/(fac_dp_scale/2),h_vals[loc_tol[dp_i][0]]/(fac_dp_scale/2)], [0,min],'r--')
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
    plt.plot(dp_list, h_vals[loc_tol[:,0]]/dp_list[:], 'r', label = 'Poly6')
    plt.plot(dp_list, h_vals[loc_tol[:,1]]/dp_list[:], 'b', label = 'Wendland')
    plt.plot(dp_list, h_vals[loc_tol[:,2]]/dp_list[:], 'g', label = 'Cubic spline')
    plt.xlabel('kh (factor of dp)')
    plt.ylabel('Distance (factor of dp)')
    plt.xticks(dp_list)
    plt.grid(b='True', which = 'major', linestyle = '-')
    plt.ticklabel_format(axis='x', scilimits =[-1,1])
    plt.savefig('loc_tol.png', dpi=600)
    plt.cla()
    plt.plot(dp_list, count_tol[:,0], 'r', label = 'Poly6')
    plt.plot(dp_list, count_tol[:,1], 'b', label = 'Wendland')
    plt.plot(dp_list, count_tol[:,2], 'g', label = 'Cubic spline')
    plt.xlabel('dp (m)')
    plt.ylabel('Number of particles for 1 per cent error')
    plt.xticks(dp_list)
    plt.ticklabel_format(axis='x', scilimits =[-1,1])
    plt.grid(b='True', which = 'major', linestyle = '-')
    plt.savefig('count_tol.png', dpi=600)

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    main()