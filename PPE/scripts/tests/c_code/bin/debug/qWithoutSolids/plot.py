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
            print(file)
    return files

files = list_csv(path)

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
data = np.loadtxt("normals.csv", delimiter=',')
plt.cla()
plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=1, scale_units='xy')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Normals Plot')
plt.savefig("normals.png")
plt.clf()

# DIVERGENCE
data = np.loadtxt("divergence.csv", delimiter=',')
plt.cla()
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=10, alpha=1)
plt.colorbar(label='Divergence')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Divergence Plot')
plt.savefig("divergence.png")
plt.clf()

# DIVERGENCE 2
data = np.loadtxt("divergence_2.csv", delimiter=',')
plt.cla()
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=10, alpha=1)
plt.colorbar(label='Divergence')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Divergence Plot')
plt.savefig("divergence_2.png")
plt.clf()

#Velocity
data = np.loadtxt("vel1.csv", delimiter=',')
plt.cla()
plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=0.01, scale_units='xy')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Ini Velocity Plot')
plt.savefig("velocity_ini.png")
plt.clf()

data = np.loadtxt("vel2.csv", delimiter=',')
plt.cla()
temp = np.max(data[:,3]**2+data[:,2]**2)
print(temp)
temp = math.sqrt(temp)
print(temp)
# plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=1/temp, scale_units='xy')
plt.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='blue', scale=0.01, scale_units='xy')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Final Velocity Plot')
plt.savefig("velocity_fin.png")




# # # Step 1: Read the data from the file
# filename = 'divergence.csv'
# data = np.loadtxt(filename, delimiter=',')

# # Step 2: Extract x, y, and divergence
# x = data[:, 0]
# y = data[:, 1]
# divergence = data[:, 2]

# # Step 3: Plot the divergence
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c=divergence, cmap='viridis', s=10, alpha=1)
# plt.colorbar(label='Divergence')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Divergence Plot')
# # plt.grid(True)
# plt.savefig("this.jpeg")

# # Step 1: Read the data from the file
# filename = 'divergence_2.csv'
# data = np.loadtxt(filename, delimiter=',')

# # Step 2: Extract x, y, and divergence
# x = data[:, 0]
# y = data[:, 1]
# divergence = data[:, 2]

# # Step 3: Plot the divergence
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c=divergence, cmap='viridis', s=10, alpha=1)
# plt.colorbar(label='Divergence')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Divergence Plot')
# # plt.grid(True)
# plt.savefig("that.jpeg")
