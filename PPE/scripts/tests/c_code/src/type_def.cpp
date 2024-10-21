#include "type_def.hpp"

constants define_constants(data_type size, data_type dp, data_type boundary_fac, int dpi)
{
    LOG(INFO)<< "Defining constants";
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
    // c.radius = 4 * dp; // kh, radius of influence
    c.dp_i = dpi;       // factor to scale the radius of influence
    c.radius = dpi * dp; // kh, radius of influence
    c.ker_fac = 4 / (M_PI * pow(c.radius, 8)); // the alpha constant in the kernel function
    return c;
}

std::ostream &operator <<(std::ostream &outs, const constants &c)
{
    outs << "Printing constants" << std::endl
         << "h: " << c.h << std::endl
         << "dp: " << c.dp << std::endl
         << "h_fac: " << c.h_fac << std::endl
         << "mass: " << c.mass << std::endl
         << "boundary_size: " << c.boundary_size << std::endl
         << "x_y_bn: " << c.x_y_bn << std::endl
         << "x_y_bp: " << c.x_y_bp << std::endl
         << "x_y_n: " << c.x_y_n << std::endl
         << "x_y_p: " << c.x_y_p << std::endl
         << "resolution: " << c.resolution << std::endl
         << "n_particles: " << c.n_particles << std::endl
         << "mid_idx: " << c.mid_idx << std::endl
         << "Eta: " << c.Eta << std::endl
         << "radius: " << c.radius << std::endl
         << "ker_fac: " << c.ker_fac << std::endl
         << "=====================\n";
    return outs;
}

void make_normals(const constants &c,
                    const MatrixXX &pos,
                    MatrixXX &normals_computed,
                    SpMatrixXX &gradient_x,
                    SpMatrixXX &gradient_y,
                    const Eigen::MatrixXi &p_type,
                    std::vector<std::vector<unsigned int>> &nearIndex,
                    const MatrixXX &density)
{
    LOG(INFO)<< "Making normals";
    // make normals
    #pragma omp parallel for
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        if (p_type(i) != 1)
        {
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (nearIndex[i][j] != i)
                // if (nearIndex[i][j] != i && p_type(nearIndex[i][j]) == 1)
                {
                    MatrixXX weight(1, 2);
                    weight.fill(0);
                    weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                    data_type temp = c.mass * (p_type(i) - p_type(nearIndex[i][j]))/ density(nearIndex[i][j]);
                    weight = weight * temp;
                    normals_computed.row(i) = normals_computed.row(i) - weight;
                }
            }
            normals_computed.row(i) = normals_computed.row(i).normalized();
            
        }
    }
}

void make_particles(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type, MatrixXX &normals)
{
    
    LOG(INFO)<< "Making particles";
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

                // make normals to the center  for the boundary particles
                if (pos(index, 0) < c.x_y_n)
                {
                    normals(index, 0) = 1;
                    // normals(index, 1) = 0;
                }
                if (pos(index, 0) > c.x_y_p)
                {
                    normals(index, 0) = -1;
                    // normals(index, 1) = 0;
                }
                if (pos(index, 1) < c.x_y_n)
                {
                    // normals(index, 0) = 0;
                    normals(index, 1) = 1;
                }
                if (pos(index, 1) > c.x_y_p)
                {
                    // normals(index, 0) = 0;
                    normals(index, 1) = -1;
                }
            }
            else
            {
                vel(index, 0) = sin(pos(index, 0)) * sin(pos(index, 0));
                vel(index, 1) = cos(pos(index, 1)) * cos(pos(index, 1));
                // if (pos(index, 0) > 0 && pos(index, 1) > 0 && pos(index, 0) < 30 * c.dp && pos(index, 1) < c.dp * 30)
                // // if (pos(index, 0) > 0 && pos(index, 1) > 0 && pos(index, 0) < c.x_y_p * 0.15 && pos(index, 1) < c.x_y_p * 0.15)
                // {
                //     vel(index, 0) = 2;
                //     vel(index, 1) = 2;
                // }
            }
        }
    }
    LOG(INFO) << "Total number of fluids: "<< p_type.sum();
    LOG(INFO)<< "total particles: (index:number) "<< index <<" or "<< c.n_particles << std::endl;
    // std::cout<< "total number of fluids: "<< p_type.sum() << std::endl;
    // std::cout<< "total particles: (index:number) "<< index <<" or "<< c.n_particles << std::endl;
}

// void logging(std::string message, int level, std::string filename ="log.txt", bool toFile = true)
// {
//     // level 0: error
//     // level 1: warning
//     // level 2: info
//     // level 3: debug
    
//     std::ofstream file;
//     file.open(filename, std::ios::out);
//     if (file.is_open())
//     {
//         file << message << std::endl;
//         file.close();
//     }
//     else
//     {
//         std::cerr << "Unable to open file " << filename << std::endl;
//     }

// }