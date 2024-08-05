#ifndef COMPUTE_HPP
#define COMPUTE_HPP
#include <math.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include "type_def.hpp"
#include "in_out.hpp"
#include "kernel.hpp"




void calc_divergence(const MatrixXX &pos,
                const MatrixXX &vel,
                const MatrixXX &density,
                const Eigen::MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<data_type>> &nearDist,
                MatrixXX &divergence,
                const Eigen::SparseMatrix<data_type> &gradient_x,
                const Eigen::SparseMatrix<data_type> &gradient_y,
                const constants &c);

void pressure_poisson(const MatrixXX &pos,
                MatrixXX &vel,
                const MatrixXX &density,
                const Eigen::MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<data_type>> &nearDist,
                MatrixXX &divergence,
                const Eigen::SparseMatrix<data_type> &gradient_x,
                const Eigen::SparseMatrix<data_type> &gradient_y,
                const Eigen::SparseMatrix<data_type> &laplacian,
                const constants &c);

MatrixXX cal_div_part_vel(const MatrixXX &pos,
                const MatrixXX &density,
                const Eigen::MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<data_type>> &nearDist,
                const MatrixXX &p,
                const Eigen::SparseMatrix<data_type> &gradient_x,
                const Eigen::SparseMatrix<data_type> &gradient_y,
                const constants &c);
#endif