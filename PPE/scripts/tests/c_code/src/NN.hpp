#include "type_def.hpp"
#include "KDTree.hpp"
#include <vector>
#include <omp.h>


void initialise_NN(constants &c,
                 std::vector<std::vector<data_type>> &pos,
                 std::vector<std::vector<unsigned int>> &nearIndex, 
                 std::vector<std::vector<double>> &nearDist);

// KDTree makeTree(unsigned int,
//                 const std::vector<dataPartDSPH> &vecDSPH,
//                 std::vector<std::vector<data_type>> &partLocations);