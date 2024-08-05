#include "NN.hpp"

void initialise_NN(constants &c,
                   MatrixXX &pos,
                   std::vector<std::vector<unsigned int>> &nearIndex,
                   std::vector<std::vector<double>> &nearDist)
{
    // auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> pos_vec(c.n_particles, std::vector<double>(2, 0));
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        pos_vec[i][0] = (double)pos(i, 0);
        pos_vec[i][1] = (double)pos(i, 1);
    }
    KDTree tree(pos_vec);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cerr << "Time for preparing the tree: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    // start = std::chrono::high_resolution_clock::now();
    std::cout<< "pos_vec.size() = " << pos_vec.size() << std::endl;

#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < pos_vec.size(); i++)
    {
        nearIndex[i] = (tree.neighborhood_indices(pos_vec[i], nearDist[i], c.radius));
    }
    // end = std::chrono::high_resolution_clock::now();
    // std::cerr << "Time for finding NN: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1e6 << " seconds\n";
}
