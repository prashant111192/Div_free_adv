#include "in_out.hpp"

void print_constants(constants c)
{
    std::cout<< "=====================\n";
    std::cout<< "Printing constants" << std::endl;
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
    std::cout << "ker_fac: " << c.ker_fac << std::endl;
    std::cout<< "=====================\n";
}

// template <typename T>
// int getDimension(const std::vector<T> &)
// {
//     return 1;
// }

// // Recursive case for higher-dimensional vectors
// template <typename T>
// int getDimension(const std::vector<std::vector<T>> &vec)
// {
//     return 1 + getDimension(vec[0]);
// }

// template <typename ele_type>
// ostream &operator<<(ostream &os, const vector<ele_type> &vect_name)
// {
//     for (auto itr : vect_name)
//     {
//         os << itr << " ";
//     }
//     return os;
// }

// template <typename T>
// void write_file(std::vector<std::vector<T>> x, string filename)
// {

//     fstream file;
//     file.open(filename, ios::out);
//     std::cout << "2D vector" << endl;
//     for (int i = 0; i < x.size(); i++)
//     {
//         for (int j = 0; j < x[i].size(); j++)
//         {
//             file << x[i][j];
//             if (j<x[i].size()-1)
//             {
//                 file << ",";
//             }
//         }
//         file << endl;
//     }
//     file.close();
// }

// template <typename T>
// void write_file(std::vector<T> x, string filename)
// {
//     fstream file;
//     file.open(filename, ios::out);
//     std::cout << "1D vector" << endl;
//     for (int i = 0; i < x.size(); i++)
//     {
//         file << x[i] << endl;
//     }
//     file.close();
// }

// template <typename T>
// void save_data(T x, string filename)
// {
//     // save the vector as a csv file
//     // x: vector to be saved, can be 1D or 2D
//     // filename: name of the file to be saved
//     int dim = getDimension(x);
//     std::cout << "Dimension: " << dim << endl;
//     if (x.size() == 0)
//     {
//         std::cout << "Empty vector" << endl;
//     }
//     else
//     {
//         std::cout<< x.size() << " value" << x[0]<< std::endl;
//         if (dim == 2)
//         {
//             write_file(x, filename);
//         }
//         if (dim == 1)
//         {
//             write_file(x, filename);
//         }
//     }
// }
