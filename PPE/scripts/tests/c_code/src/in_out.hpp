#ifndef IN_OUT_HPP
#define IN_OUT_HPP
#include <fstream>
#include <iostream>
#include <string>

#include "type_def.hpp"

template <typename T>
void writeMatrixToFile(const MatrixXX &pos, T x, std::string filename);



void print_constants(constants c);

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
//     int rows, cols, dim;
//     rows = x.rows();
//     cols = x.cols();
//     if rows ==1 || cols == 1
//     {
//         std::cout << "1D vector" << endl;
//         dim = 1;
//     }
//     else
//     {
//         std::cout << "2D vector" << endl;
//         dim = 2;
//     }
//     if (dim == 2)
//     {
//         std::cout << x.rows() << " rows" << x.cols() << " cols" << std::endl;
//         write_file(x, filename);
//     }else
//     {
//         std::cout << x.size() << " elements" << std::endl;
//         write_file(x, filename);
//     }
// }

// using namespace std;
// template <typename T>
// int getDimension(const std::vector<T> &);

// template <typename T>
// int getDimension(const std::vector<std::vector<T>> &vec);

// template <typename ele_type>
// ostream &operator<<(ostream &os, const vector<ele_type> &vect_name);

// template <typename T>
// void write_file(std::vector<std::vector<T>> x, string filename);

// template <typename T>
// void write_file(std::vector<T> x, string filename);

// template <typename T>
// void save_data(T x, string filename);
#endif