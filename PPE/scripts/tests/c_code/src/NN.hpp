#ifndef NN_HPP
#define NN_HPP
#include "type_def.hpp"
#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>

void initialise_NN(constants &c,
                 MatrixXX &pos,
                 std::vector<std::vector<unsigned int>> &nearIndex, 
                 std::vector<std::vector<double>> &nearDist);



data_type calcualte_distance(MatrixXX p1, MatrixXX p2);

// void Print_Verlet_List(std::vector<std::vector<unsigned int>> verlet_list)
// {
//     for (unsigned int i = 0; i < verlet_list.size(); i++)
//     {
//         std::cout << "Bin " << i << " contains particles: ";
//         for (unsigned int j = 0; j < verlet_list[i].size(); j++)
//         {
//             std::cout << verlet_list[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }


// void Print_nn_list(std::vector<std::vector<unsigned int>> nn_list, std::vector<Particle> points)
// {
//     // Printing out the neighbour list and the distance between the neighbouring particles
//     unsigned count = 0;
//     for (unsigned int i = 0; i < nn_list.size(); i++)
//     {
//         std::cout << "Particle " << i << " has neighbours: ";
//         for (unsigned int j = 0; j < nn_list[i].size(); j++)
//         {
//             std::cout << nn_list[i][j] << " ";
//             std::cout << "Distance between the particles: " << Distance(points[i], points[nn_list[i][j]]) << std::endl;
//             count++;
//         }
//     }
//     std::cout << "Total number of neighbours: " << count << std::endl;
// }


#endif