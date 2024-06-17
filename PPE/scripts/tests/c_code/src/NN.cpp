#include "NN.hpp"

KDTree makeTree(long unsigned int NParticle,
                const std::vector<dataPartDSPH> &vecDSPH,
                std::vector<std::vector<data_type>> &partLocations)
{
    for (long unsigned i = 0; i < NParticle; i++) {
        partLocations[i] = {vecDSPH[i].DSPH[0],
                            vecDSPH[i].DSPH[1],
                            vecDSPH[i].DSPH[2]};
    }
    KDTree tree(partLocations);
    return tree;
}

void initialise_NN(constants &c,
                 std::vector<std::vector<data_type>> &pos,
                 std::vector<std::vector<long unsigned int>> &nearIndex, 
                 std::vector<std::vector<double>> &distPoint)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<data_type>> partLocations(c.n_particles, std::vector<data_type>(3,0));
    std::vector<std::vector<float>> test(100, std::vector<float>(3,0)); 
    KDTree tree(test);

    end_time = time(NULL);
    // std::cout << "\ttime for preparing the tree: " << end_time - start_time << "\n";

    start_time = time(NULL);
    long unsigned xxx = partLocations.size();
    #pragma omp parallel for num_threads(10)
    for (long unsigned int i = 0; i< partLocations.size(); i++) {
        // tree record all coordinates, pass particleFluid coordinates for fluid paritcles only
        nearIndex[i] = (tree.neighborhood_indices(partLocations[i], distPoint[i], h_rad));
        // int xxxxxx = nearIndex[i].size();
        // std::cout<< xxxxxx;
    }
    end_time = time(NULL);
    // std::cout << "\ttime for nearest neigbour search: " << end_time - start_time << "\n";
    double temp = 0;
    double temp2 = 0;
    for (long unsigned i = 0; i < nearIndex.size(); i++)
    {
        temp += nearIndex[i].size();
        temp2 += distPoint[i].size();
    }
    temp = temp / nearIndex.size();
    // std::cout << "average number of nn are: " << temp << std::endl;
    temp2 = temp2 / nearIndex.size();
    // std::cout << "average number of nn are: " << temp2 << std::endl;
}
// LEGACY ============================================================================================
// KDTree makeTree(long unsigned int NParticle, std::vector<adm> &myADM, std::vector<std::vector<data_type>> &partLocations)
// {
//     for (long unsigned i = 0; i < NParticle; i++)
//     {
//         partLocations.push_back(myADM[i].getPos());
//     }
//     KDTree tree(partLocations);
//     return tree;
// }

// void radNNsearch(long unsigned int NPart, std::vector<adm> &myADM, data_type h_rad, std::vector<std::vector<long unsigned int>> &nearIndex, std::vector<std::vector<double>> &distPoint)
// {

//     time_t start_time, end_time;
//     start_time = time(NULL);
//     KDTree tree;
//     std::vector<std::vector<data_type>> partLocations(NPart, std::vector<data_type>(3));
//     tree = makeTree(NPart, myADM, partLocations);

//     end_time = time(NULL);
//     std::cout << "\ntime for preparing the tree: " << end_time - start_time << "\n";
//     // std::vector<std::vector<long unsigned int>> nearIndex;

//     start_time = time(NULL);
//     #pragma omp parallel for
//     // for (long unsigned int i = 0; i < 1; i++)
//     for (long unsigned int i = 0; i< partLocations.size(); i++)
//     {
//         // nearIndex[i] = tree.neighborhood_indices(partLocations[i], h_rad);
//         nearIndex[i] = (tree.neighborhood_indices(partLocations[i], distPoint[i],h_rad));
//         // for(int l = 0; l< nearIndex[i].size(); l++)
//         // {
//         //     double temp1 = 0;
//         //     double temp2 = 0;
//         //     for (int j = 0; j < 3; j++)
//         //     {
//         //         temp2 = partLocations[i][j] - myADM[nearIndex[i][l]].getPos()[j];
//         //         temp2 = temp2 * temp2;
//         //         temp1 = temp1 + temp2;
//         //     }
//         //     temp1 = sqrt(temp1);
//         //     std::cout << temp1 << std::endl;
//         // }
//     }
//     end_time = time(NULL);
//     std::cout << "\ntime for nearest neigbour search: " << end_time - start_time << "\n";
//     double temp = 0;
//     double temp2 = 0;
//     for (long unsigned i = 0; i < nearIndex.size(); i++)
//     {
//         temp += nearIndex[i].size();
//         temp2 += distPoint[i].size();
//     }
//     temp = temp / nearIndex.size();
//     std::cout << "average number of nn are: " << temp << std::endl;
//     temp2 = temp2 / nearIndex.size();
//     std::cout << "average number of nn are: " << temp2 << std::endl;
// }