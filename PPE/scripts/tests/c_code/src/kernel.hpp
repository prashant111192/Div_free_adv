#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <math.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include "type_def.hpp"
#include "in_out.hpp"
MatrixXX gradient_poly6(const data_type &distance,
                        const constants &c,
                        const MatrixXX &r_ij);

data_type lap_poly6(const data_type distance,
                    const constants &c);

void prepare_grad_lap_matrix(const MatrixXX &pos,
                            const std::vector<std::vector<unsigned int>> &nearIndex,
                            const std::vector<std::vector<data_type>> &nearDist,
                            const constants &c,
                            SpMatrixXX &gradient_x,
                            SpMatrixXX &gradient_y,
                            SpMatrixXX &laplacian);
#endif