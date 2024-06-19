#ifndef COMPUTE_HPP
#define COMPUTE_HPP
#include <math.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include "type_def.hpp"
#include "in_out.hpp"



MatrixXX gradient_poly6(const data_type &distance, const data_type kh, const MatrixXX &r_ij);

data_type lap_poly6(const data_type distance,
                    const data_type kh);

void calc_divergence(const MatrixXX &pos,
                MatrixXX &vel,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                MatrixXX &divergence,
                const constants &c);

void pressure_poisson(const MatrixXX &pos,
                MatrixXX &vel,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                MatrixXX &divergence,
                const constants &c);

MatrixXX cal_div_part_vel(const MatrixXX &pos,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                const MatrixXX &p,
                const constants &c);
#endif