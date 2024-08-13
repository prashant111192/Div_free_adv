#include "type_def.hpp"

constants define_constants(data_type size, data_type dp, data_type boundary_fac)
{
    constants c;
    c.h = 0.02;            // why was this used??
    c.dp = dp;
    c.h_fac = c.h / c.dp; // originally used to scacle h with dp

    c.mass = 1000 * c.dp * c.dp; // mass 
    c.boundary_size = boundary_fac * c.dp;
    c.x_y_bn = -size / 2 - (c.boundary_size) / 2;
    c.x_y_bp = size / 2 + c.boundary_size / 2;
    c.x_y_n = -size / 2;
    c.x_y_p = size / 2;
    c.resolution = int((size + c.boundary_size) / c.dp);    // number of particles along one edge
    c.n_particles = c.resolution * c.resolution;
    c.mid_idx = (int)((c.n_particles) / 2);
    c.Eta = 1e-12;
    c.radius = 3 * dp; // kh, radius of influence
    c.ker_fac = 4 / (M_PI * pow(c.radius, 8));

    return c;
}

void make_particles(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type)
{
    p_type.fill(1); // all are fluid
    density.fill(1000); // density of water

    unsigned int index;
    for (unsigned int i = 0; i < c.resolution; i++)
    {
        for (unsigned int j = 0; j < c.resolution; j++)
        {
            index = i * c.resolution + j;
            pos(index, 0) = c.x_y_bn + (c.dp * i);
            pos(index, 1) = c.x_y_bn + (c.dp * j);

            if (pos(index, 0) < c.x_y_n || pos(index, 0) > c.x_y_p || pos(index, 1) < c.x_y_n || pos(index, 1) > c.x_y_p)
            {
                p_type(index) = 0; // p_type ==0 =>Boundary particle
            }
            if (pos(index, 0) > 0 && pos(index, 1) > 0 && pos(index, 0) < c.x_y_p * 0.15 && pos(index, 1) < c.x_y_p * 0.15)
            {
                vel(index, 0) = 0.01;
                vel(index, 1) = 0.01;
            }
        }
    }
    std::cout<< "total number of fluids: "<< p_type.sum() << std::endl;
    std::cout<< "total particles: (index:number) "<< index <<" or "<< c.n_particles << std::endl;
}