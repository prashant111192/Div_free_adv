#ifndef DIV_DIFF_HPP
#define DIV_DIFF_HPP
#include <math.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>

#include "type_def.hpp"
#include "in_out.hpp"
#include "kernel.hpp"
void div_diff_compute(const MatrixXX &pos,
                      MatrixXX &vel,
                      const MatrixXX &density,
                      const Eigen::MatrixXi &p_type,
                      const std::vector<std::vector<unsigned int>> &nearIndex,
                      const std::vector<std::vector<data_type>> &nearDist,
                      MatrixXX &divergence,
                      const Eigen::SparseMatrix<data_type> &gradient_x,
                      const Eigen::SparseMatrix<data_type> &gradient_y,
                      const Eigen::SparseMatrix<data_type> &laplacian,
                      const constants &c,
                      const unsigned int count);
#endif