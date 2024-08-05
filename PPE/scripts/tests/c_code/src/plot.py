import numpy as np
import matplotlib.pyplot as plt
import glob

def list_files(directory = './', extension= 'csv'):
    return glob.glob(f'{directory}/*vel.{extension}')

def read_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    return data

def main():
    pos = read_data('pos.csv')
    divergence = read_data('divergence.csv')
    p_type = read_data('p_type.csv')
    vel_file = list_files()
    plt.cla()
    plt.scatter(pos[:,0], pos[:,1], c=divergence, cmap='coolwarm')
    plt.colorbar()
    plt.savefig('divergence.png', dpi=600)
    plt.clf()
    plt.cla()
    plt.scatter(pos[:,0], pos[:,1], c=p_type, cmap='coolwarm', s=0.01, alpha=1)
    # plt.scatter(pos[::5,0], pos[::5,1], c=p_type[::5], cmap='coolwarm')
    plt.colorbar()
    plt.savefig('p_type.png', dpi=600)
    for file_name in vel_file:
        vel = read_data(file_name)
        plt.clf()
        plt.cla()
        plt.quiver(pos[::27,0], pos[::27,1], vel[::27,0], vel[::27,1], scale=1, scale_units='xy')
        plt.savefig(f'{file_name}.png', dpi=600)





if __name__ == '__main__':
    main()