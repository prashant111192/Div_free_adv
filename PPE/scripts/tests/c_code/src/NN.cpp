#include "NN.hpp"

void initialise_NN(constants &c,
                   std::vector<std::vector<data_type>> &pos,
                   std::vector<std::vector<unsigned int>> &nearIndex,
                   std::vector<std::vector<double>> &nearDist)
{
    auto start = std::chrono::high_resolution_clock::now();
    KDTree tree(pos);
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "Time for preparing the tree: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    std::cout<< "pos.size() = " << pos.size() << std::endl;
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < pos.size(); i++)
    {
        // tree record all coordinates, pass particleFluid coordinates for fluid paritcles only
        // std::cout << "pos[i].size() = " << pos[i].size() << std::endl;
        nearIndex[i] = (tree.neighborhood_indices(pos[i], nearDist[i], c.radius));
        // std::cout << "nearIndex[i].size() = " << nearIndex[i].size() << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    std::cerr << "Time for finding NN: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1e6 << "ms\n";
    // std::cout << "\ttime for nearest neigbour search: " << end_time - start_time << "\n";
}
