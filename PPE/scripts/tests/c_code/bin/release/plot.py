import matplotlib.pyplot as plt
import numpy as np
import os
import math
path = './'
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

def list_csv(path):
    this = os.listdir(path)
    files = []
    for file in (this):
        if file.endswith(".csv"):
            files.append(file)
    return files

files = list_csv(path)
print(files)
files.sort()
print(files)

#out.txt
# name = "out.txt"
# f = open(name, 'r')
# count = 0
# data = []
# for x in f:
#     if count<66:
#         count += 1
#         continue
#     x = x.split(";")
#     x = x[1].split(" ")
#     x = x[-1]
#     x = x.replace("\n", "")
#     data.append(x)
#     count += 1
# data = np.array(data)
# data = data.astype(np.float64)
# plt.cla()
# plt.plot(data)
# plt.title('Error vs Iteration Plot')
# plt.xlabel('Iteration')
# plt.ylabel('Error')
# plt.grid(True)
# plt.savefig("error.png")
# plt.clf()

# exit()


# NORMALS
if "normals.csv" in files:
    data = np.loadtxt("normals.csv", delimiter=',')
    plt.cla()
    plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=1, scale_units='xy')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Normals Plot')
    plt.savefig("normals.png")
    plt.clf()

    # DIVERGENCE
print("Making figures")
for file in files:
    print('.', end='')
    name = file.replace(".csv", ".png")
    if "divergence" in file:
        data = np.loadtxt(file, delimiter=',')
        plt.cla()
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=10, alpha=1)
        plt.colorbar(label='Divergence')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{name} Plot')
        plt.savefig(file.replace(".csv", ".png"))
        plt.clf()
    
    # VELOCITY
    elif "vel" in file:
        data = np.loadtxt(file, delimiter=',')
        plt.cla()
        # temp = np.max(data[:,3]**2+data[:,2]**2)
        # temp = math.sqrt(temp)
        plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=1, scale_units='xy')
        # plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=1/temp, scale_units='xy')
        plt.gca().set_aspect('equal', adjustable='box')
        # name = file.replace(".csv", ".png")
        plt.title(f'{name} Plot')
        plt.savefig(file.replace(".csv", ".png"))
        plt.clf()

log = open("data.out", w)