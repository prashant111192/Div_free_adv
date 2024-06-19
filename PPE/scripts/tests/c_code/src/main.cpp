#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>


#include "type_def.hpp"
#include "in_out.hpp"
#include "NN.hpp"
#include "compute.hpp"

using namespace std;

constants define_constants(data_type size, data_type dp, data_type boundary_fac)
{
    constants c;
    c.h = 0.02;
    c.dp = dp;
    c.h_fac = c.h / c.dp; // originally used to scacle h with dp
    c.mass = 1000 * c.dp * c.dp; // mass 
    c.boundary_size = boundary_fac * c.dp;
    c.x_y_bn = -size / 2 - (c.boundary_size) / 2;
    c.x_y_bp = size / 2 + c.boundary_size / 2;
    c.x_y_n = -size / 2;
    c.x_y_p = size / 2;
    c.resolution = int((size + c.boundary_size) / c.dp);
    c.n_particles = c.resolution * c.resolution;
    c.mid_idx = (int)((c.n_particles) / 2);
    c.Eta = 1e-12;
    c.radius = 12 * dp; // kh, radius of influence

    return c;
}

void make_particles(const constants &c, vector<vector<data_type>> &pos, vector<vector<data_type>> &vel, vector<data_type> &density, vector<int> &p_type)
{
    for (unsigned int i = 0; i < c.resolution; i++)
    {
        for (unsigned int j = 0; j < c.resolution; j++)
        {
            unsigned int index = i * c.resolution + j;
            pos[index] = {c.x_y_bn + (c.dp * i), c.x_y_bn + (c.dp * j)};

            if (pos[index][0] < c.x_y_n || pos[index][0] > c.x_y_p || pos[index][1] < c.x_y_n || pos[index][1] > c.x_y_p)
            {
                p_type[index] = 0; // p_type ==0 =>Boundary particle
            }
            if (pos[index][0] > 0 && pos[index][1] > 0 && pos[index][0] < c.x_y_p * 0.5 && pos[index][1] < c.x_y_p * 0.5)
            {
                vel[index] = {0.05, 0.05};
            }
        }
    }
}
 void print_constants(constants c)
 {
        std::cout << "h: " << c.h << std::endl;
        std::cout << "dp: " << c.dp << std::endl;
        std::cout << "h_fac: " << c.h_fac << std::endl;
        std::cout << "mass: " << c.mass << std::endl;
        std::cout << "boundary_size: " << c.boundary_size << std::endl;
        std::cout << "x_y_bn: " << c.x_y_bn << std::endl;
        std::cout << "x_y_bp: " << c.x_y_bp << std::endl;
        std::cout << "x_y_n: " << c.x_y_n << std::endl;
        std::cout << "x_y_p: " << c.x_y_p << std::endl;
        std::cout << "resolution: " << c.resolution << std::endl;
        std::cout << "n_particles: " << c.n_particles << std::endl;
        std::cout << "mid_idx: " << c.mid_idx << std::endl;
        std::cout << "Eta: " << c.Eta << std::endl;
        std::cout << "radius: " << c.radius << std::endl;
 }

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Let's do this!!" << std::endl;
    data_type size = 1;
    data_type dp = 0.001;
    auto boundary_fac = 12;
    constants c = define_constants(size, dp, boundary_fac);

    print_constants(c);
    std::vector<std::vector<data_type>> pos(c.n_particles, std::vector<data_type>(2, 0));
    std::vector<std::vector<data_type>> vel(c.n_particles, std::vector<data_type>(2, 0));
    vector<data_type> density(c.n_particles, 1000);
    vector<int> p_type(c.n_particles, 1);
    data_type mass = density[0] * dp * dp;
    make_particles(c, pos, vel, density, p_type);
    std::vector<std::vector<double>> nearDist(c.n_particles);         // [center particle, neighbor particles] generated from vecDSPH with correspongding idx
    std::vector<std::vector<unsigned>> nearIndex(c.n_particles); // [center particle, neighbor particles] generated from vecDSPH with correspongding idx
    initialise_NN(c, pos, nearIndex, nearDist);
    int count = 0;
    for (unsigned int j = 0; j<nearIndex.size(); j++)
    {
        if (count < nearIndex[j].size())
        {
            count = nearIndex[j].size();
        }
    }
    std::cout << "Maximum number of NN: "<< count << std::endl;
    // DIVERGENCE
    std::vector<data_type> divergence(c.n_particles, 0);
    calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, c);

    save_data(p_type, "p_type.csv");
    save_data(divergence, "divergence.csv");
    save_data(pos, "pos.csv");
    save_data(vel, "vel.csv");
    std::cout << "resolution: " << c.resolution << std::endl;
    std::cout << "n_particles: " << c.n_particles << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);\
    size_t total_NN;
    size_t NN_i= 0;
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        NN_i += nearIndex[i].capacity();
    }
    total_NN = NN_i * sizeof(unsigned int);
    std::cout << "Total memory used for NN_index: " << total_NN/(1024*1024) << " Mbytes" << std::endl;
    total_NN = NN_i * sizeof(data_type);
    std::cout << "Total memory used for NN_distance: " << total_NN/(1024*1024) << " Mbytes" << std::endl;
    std::cout<< "size of double: " << sizeof(data_type) << "and size of unsigned int: " << sizeof(unsigned int) << std::endl;
    std::cout << "Time taken by function: "
         << duration.count()/1e6 << " seconds" << std::endl;

    return 0;
}