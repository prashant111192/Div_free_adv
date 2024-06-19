#include <math.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include "type_def.hpp"


void calc_divergence(const std::vector<std::vector<data_type>> &pos,
                const std::vector<std::vector<data_type>> &vel,
                const std::vector<data_type> &density,
                const std::vector<int> &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                std::vector<data_type> &divergence,
                const constants &c);