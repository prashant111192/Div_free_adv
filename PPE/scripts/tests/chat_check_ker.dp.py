import matplotlib.pyplot as plt
import sklearn.neighbors as NN
import numpy as np

def make_particles(length=1, boundary_fac=40, dp=0.008):
    boundary_width = dp * boundary_fac
    x1, x2 = -length / 2 - boundary_width / 2, length / 2 + boundary_width / 2
    y1, y2 = -length / 2 - boundary_width / 2, length / 2 + boundary_width / 2
    resolution = int((length + boundary_width) / dp)
    
    positions = np.zeros((resolution ** 2, 2))
    density = np.ones(resolution ** 2) * 1000
    velocity = np.zeros((resolution ** 2, 2))
    mass = density[1] * dp ** 2
    type = np.zeros(resolution ** 2)
    
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            positions[count] = np.array([x1 + (dp * i), y1 + (dp * j)])
            if x1 + boundary_width / 2 < (x1 + (dp * i)) < x2 - dp - boundary_width / 2 and y1 + boundary_width / 2 < (y1 + (dp * j)) < y2 - dp - boundary_width / 2:
                type[count] = 1
            count += 1

    h_const = 0.02
    dp_const = 0.008
    h_fac = h_const / dp_const
    h = h_fac * dp

    kh = h * 2
    temp = int((resolution * resolution / 2))

    return positions, velocity, density, mass, type, kh, h, temp

def poly6(kh, distance):
    h = kh / 2
    h1 = 1 / h
    q = distance * h1
    fac = (4 * h1 ** 8) / np.pi
    temp = h ** 2 - distance ** 2
    return fac * (temp) ** 3 if 0 < distance < h else 0

def get_fac_w(h1):
    fac = 7 / (4 * np.pi)
    return fac * h1 ** 2

def ker_w(kh, distance):
    h = kh / 2
    h1 = 1 / h
    q = distance * h1
    temp = 1 - 0.5 * q
    fac = get_fac_w(h1)
    return fac * temp ** 4 * (2 * q + 1) if 0 < q <= 2 else 0

def get_fac_cs(h1):
    fac = 10 / (7 * np.pi)
    return fac * h1 ** 2

def ker_cs(kh, distance):
    h = kh / 2
    h1 = 1 / h
    q = distance * h1
    fac = get_fac_cs(h1)
    tmp2 = 2. - q
    if q > 2.0:
        val = 0.0
    elif q > 1.0:
        val = 0.25 * tmp2 ** 3
    else:
        val = 1 - 1.5 * q ** 2 * (1 - 0.5 * q)
    return val * fac

def check_kernel_summation_for_particle(i, positions, kh, NN_idx, mass):
    den = np.zeros(3)
    for j in NN_idx[0]:
        r_ij = positions[i] - positions[j]
        distance = np.linalg.norm(r_ij)
        if distance <= kh and distance > 0.0:
            den[0] += poly6(kh, distance) * mass
            den[1] += ker_w(kh, distance) * mass
            den[2] += ker_cs(kh, distance) * mass
    return den

def check_kernel_summation(positions, kh, NN_idx, mass):
    summation = np.zeros((len(positions), 3))
    for i in range(len(positions)):
        summation[i] = check_kernel_summation_for_particle(i, positions, kh, NN_idx, mass)
    return summation

def plot_prop(positions, prop, title, climax=None, climin=None):
    plt.clf()
    plt.scatter(positions[:, 0], positions[:, 1], c=prop, cmap='viridis', s=2)
    if climax is not None:
        plt.clim(vmax=climax, vmin=climin)
    plt.colorbar()
    plt.savefig(f'{title}.png')

def plot_NN_grp(positions, NN_idx, title, mid, h_vals, loc_tol, ker_vals, dp, boundary_fac, kh_dsph):
    plt.cla()
    x1, x2 = np.min(positions[:, 0]), np.max(positions[:, 0])
    y1, y2 = np.min(positions[:, 1]), np.max(positions[:, 1])
    r_min, r_max = np.min(h_vals), np.max(h_vals)
    
    idx = mid
    circle1 = plt.Circle((positions[idx, 0], positions[idx, 1]), kh_dsph, color='k', fill=False, linewidth=0.5)
    circle_poly = plt.Circle((positions[idx, 0], positions[idx, 1]), h_vals[loc_tol[0]], color='r', linestyle='--', fill=False, linewidth=0.5)
    circle_w = plt.Circle((positions[idx, 0], positions[idx, 1]), h_vals[loc_tol[1]], color='b', linestyle='--', fill=False, linewidth=0.5)
    circle_cs = plt.Circle((positions[idx, 0], positions[idx, 1]), h_vals[loc_tol[2]], color='g', linestyle='--', fill=False, linewidth=0.5)

    plt.plot([x1, x2], [y1, y1], 'k-')
    plt.plot([x1, x2], [y2, y2], 'k-')
    plt.plot([x1, x1], [y1, y2], 'k-')
    plt.plot([x2, x2], [y1, y2], 'k-')
    x1 += dp * boundary_fac / 2
    x2 -= dp * boundary_fac / 2
    y1 += dp * boundary_fac / 2
    y2 -= dp * boundary_fac / 2
    plt.plot([x1, x2], [y1, y1], 'b--')
    plt.plot([x1, x2], [y2, y2], 'b--')
    plt.plot([x1, x1], [y1, y2], 'b--')
    plt.plot([x2, x2], [y1, y2], 'b--')

    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle_poly)
    plt.gca().add_patch(circle_w)
    plt.gca().add_patch(circle_cs)

    plt.xlim(x1 - r_max, x2 + r_max)
    plt.ylim(y1 - r_max, y2 + r_max)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(positions[NN_idx[0], 0], positions[NN_idx[0], 1], s=0.5, alpha=0.5)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.savefig(f'NN_grp_{title}.png')

def process_dp(dp, length, boundary_fac):
    print(f'dp: {dp}')
    pos, vel, density, mass, type, kh, h, mid = make_particles(length, boundary_fac, dp)
    radius_ = length / 2
    center_pt = np.array([0, 0])
    nbrs = NN.NearestNeighbors(radius=radius_, algorithm='kd_tree').fit(pos)
    closest_to_center = nbrs.kneighbors([center_pt], 1)
    closest_index = closest_to_center[1].flatten()[0]
    closest_distance = closest_to_center[0].flatten()[0]
    print(f'closest_to_center particle: {closest_index} at distance: {closest_distance}')
    NN_idx = nbrs.radius_neighbors(pos[closest_index].reshape(1, -1))[1].flatten()
    print(f'Number of neighbors of closest particle: {len(NN_idx[0])}')
    
    h_vals = np.linspace(1.5 * h, radius_, 1000)
    ker_vals = np.zeros((len(h_vals), 3))
    
    for i, h_val in enumerate(h_vals):
        kh_ = h_val
        ker_vals[i, :] = check_kernel_summation_for_particle(closest_index, pos, kh_, NN_idx, mass)
    
    loc_tol = np.zeros(3, dtype=int)
    for j in range(3):
        try:
            loc_tol[j] = int(np.argwhere(ker_vals[:, j] > 900)[0][0])
        except:
            loc_tol[j] = 0

    plot_NN_grp(pos, NN_idx, f'{dp}', closest_index, h_vals, loc_tol, ker_vals, dp, boundary_fac, kh)
    
    plt.cla()
    fac_dp_scale = dp
    plt.plot(h_vals / fac_dp_scale, (1000 - ker_vals[:, 0]) / 10, 'r', label='Poly6')
    plt.plot(h_vals / fac_dp_scale, (1000 - ker_vals[:, 1]) / 10, 'b', label='Wendland')
    plt.plot(h_vals / fac_dp_scale, (1000 - ker_vals[:, 2]) / 10, 'g', label='Cubic spline')
    min_val = np.max(1000 - ker_vals[:, 0]) / 10
    plt.plot([kh / fac_dp_scale, kh / fac_dp_scale], [0, min_val], 'k--', label='DSPH_default')
    plt.plot([h_vals[loc_tol[0]] / fac_dp_scale, h_vals[loc_tol[0]] / fac_dp_scale], [0, min_val], 'r--')
    plt.plot([h_vals[loc_tol[1]] / fac_dp_scale, h_vals[loc_tol[1]] / fac_dp_scale], [0, min_val], 'b--')
    plt.plot([h_vals[loc_tol[2]] / fac_dp_scale, h_vals[loc_tol[2]] / fac_dp_scale], [0, min_val], 'g--')
    plt.xlabel('kh (factor of dp)')
    plt.ylabel('Density, reference density = 1000')
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.5)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.legend(loc='upper right')
    plt.savefig(f'ker_vals_{dp}.png')
    plt.ylim(0, 1)
    plt.savefig(f'ker_vals_zoom_{dp}.png')
    return h_vals[loc_tol] / dp

def main():
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    length = 1
    boundary_fac = 40
    dp_list = np.linspace(0.02, 0.04, 4)[::-1]
    dp_list = np.round(dp_list, 4)
    loc_tol_all = np.zeros((len(dp_list), 3))

    for dp_i, dp in enumerate(dp_list):
        loc_tol_all[dp_i] = process_dp(dp, length, boundary_fac)
    
    plt.cla()
    plt.plot(dp_list, loc_tol_all[:, 0], 'r', label='Poly6')
    plt.plot(dp_list, loc_tol_all[:, 1], 'b', label='Wendland')
    plt.plot(dp_list, loc_tol_all[:, 2], 'g', label='Cubic spline')
    plt.xlabel('kh (factor of dp)')
    plt.ylabel('Distance (factor of dp)')
    plt.legend()
    plt.xticks(dp_list)
    plt.ticklabel_format(axis='x', scilimits=[-1, 1])
    plt.savefig('loc_tol.png', dpi=600)

if __name__ == "__main__":
    main()
