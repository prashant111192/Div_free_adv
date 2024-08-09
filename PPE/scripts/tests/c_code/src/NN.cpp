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
    std::cout<<" pos_vec[0][0] = " << pos_vec[0][0] << std::endl;
    std::cout<<" pos(0,0), pos(0,1) = " << pos(0,0)<< "," << pos(0,1) << std::endl;
    std::cout<<"  = " << pos(0,0)<< "," << pos(0,1) << std::endl;
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
void print_nodes(const Kdtree::KdNodeVector &nodes) {
  size_t i,j;
  for (i = 0; i < nodes.size(); ++i) {
    if (i > 0)
      std::cout << " ";
    std::cout << "(";
    for (j = 0; j < nodes[i].point.size(); j++) {
      if (j > 0)
        std::cout << ",";
      std::cout << nodes[i].point[j];
    }
    std::cout << ")";
  }
  std::cout << " : ";
}

void print_nodes_distance(const Kdtree::KdNodeVector &nodes, std::vector<double> point) {
  size_t i,j;
  for (i = 0; i < nodes.size(); ++i) {
    if (i > 0)
      std::cout << " ";
    std::cout << "(";
    for (j = 0; j < nodes[i].point.size(); j++) {
      if (j > 0)
        std::cout << ",";
      std::cout << nodes[i].point[j];
    }
    std::cout << ")";
  }
  std::cout << " : ";
}
void initialise_NN_2(constants &c,
                   MatrixXX &pos,
                   std::vector<std::vector<unsigned int>> &nearIndex,
                   std::vector<std::vector<double>> &nearDist)
{
    // auto start = std::chrono::high_resolution_clock::now();
    Kdtree::KdNodeVector nodes;
    std::vector<std::vector<double>> pos_vec(c.n_particles, std::vector<double>(2, 0));
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        std::vector<data_type> point(2);
        pos_vec[i][0] = (double)pos(i, 0);
        pos_vec[i][1] = (double)pos(i, 1);
        point[0] = pos(i, 0);
        point[1] = pos(i, 1);
        nodes.push_back(Kdtree::KdNode(point));
        // nodes.push_back(Kdtree::KdNode(pos_vec[i]));
    }
    // for (int i = 0; i< c.n_particles; i++)
    // {
    //     std::vector<data_type> point(2);
        // point[0] = pos(i, 0);
        // point[1] = pos(i, 1);
        // nodes.push_back(Kdtree::KdNode(point));
    // }

    Kdtree::KdTree tree(&nodes, 0);
    std::vector<Kdtree::KdNodeVector> ss;
    unsigned int count= 0;
    for (int i = 0; i < c.n_particles; i++)
    {
        Kdtree::KdNodeVector nn;
        std::vector<data_type> point(2);
        point[0] = pos(i, 0);
        point[1] = pos(i, 1);
        tree.range_nearest_neighbors(point, c.radius, &nn);
        ss.push_back(nn);
        count = count + nn.size();
        if (i == 0)
        {
            std::cout << "NN for particle 0: ";
            print_nodes(nn);
            std::cout << std::endl;
        }
    }
    std::cout << "Total number of neighbors using the new NN when calculated while making: " << count << std::endl;
    count = 0;
    for (int i = 0; i < c.n_particles; i++)
    {
        count= count+ ss[i].size();
    }
    std::cout << "Total number of neighbors using the new NN: " << count << std::endl;

}