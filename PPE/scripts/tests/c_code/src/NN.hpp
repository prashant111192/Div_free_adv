#include "type_def.hpp"
#include "KDTree.hpp"
#include <vector>

void print();

void initialise_NN(constants &c,
                 std::vector<std::vector<data_type>> &pos,
                 std::vector<std::vector<long unsigned int>> &nearIndex, 
                 std::vector<std::vector<double>> &distPoint);

// KDTree makeTree(long unsigned int,
//                 const std::vector<dataPartDSPH> &vecDSPH,
//                 std::vector<std::vector<data_type>> &partLocations);