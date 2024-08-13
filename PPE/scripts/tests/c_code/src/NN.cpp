#include "NN.hpp"

data_type calcualte_distance(MatrixXX p1, MatrixXX p2)
{
  data_type dist = 0;
  for (unsigned int i = 0; i < p1.size(); i++)
  {
    dist += (p1(i) - p2(i)) * (p1(i) - p2(i));
  }
  return std::sqrt(dist);
}

void initialise_NN(constants &c,
                   MatrixXX &pos,
                   std::vector<std::vector<unsigned int>> &nearIndex,
                   std::vector<std::vector<double>> &nearDist)
{
  data_type bin_size = c.radius * 2;
  int n_bins_x, n_bins_y;
  if (c.x_y_bn < 0)
  {
    n_bins_x = 4 + (c.x_y_bp + abs(c.x_y_bn)) / bin_size;
  }
  else
  {
    n_bins_x = 4 + (c.x_y_bp) / bin_size;
  }
  if (c.x_y_bn < 0)
  {
    n_bins_y = 4 + (c.x_y_bp + abs(c.x_y_bn)) / bin_size;
  }
  else
  {
    n_bins_y = 4 + (c.x_y_bp) / bin_size;
  }
  int total_bins = n_bins_x * n_bins_y;

  std::vector<std::vector<unsigned int>> verlet_list(total_bins);

  for (unsigned int i = 0; i < c.n_particles; i++)
  {
    data_type actual_x = pos(i, 0);
    data_type actual_y = pos(i, 1); 
    data_type x = pos(i, 0) - c.x_y_bn + (2 * bin_size);
    data_type y = pos(i, 1) - c.x_y_bn + (2 * bin_size);

    unsigned bin_x = x / bin_size;
    unsigned bin_y = y / bin_size;
    unsigned bin = bin_x + n_bins_x * bin_y;
    verlet_list[bin].push_back(i);
  }
  
  
  for (unsigned int i = 0; i< total_bins; i++)
  {
  #pragma omp parallel for num_threads(10)
    for (unsigned int j = 0; j< verlet_list[i].size(); j++)
    {
      unsigned current_part_idx = verlet_list[i][j];

      for (int x_itr = -1; x_itr <= 1; x_itr++)
      {
        for (int y_itr = -1; y_itr <= 1; y_itr++)
        {
          int neighbour_bin_x = (i % n_bins_x) + x_itr;
          int neighbour_bin_y = (i / n_bins_x) + y_itr;

          if (neighbour_bin_x < 0 || neighbour_bin_x >= n_bins_x ||
              neighbour_bin_y < 0 || neighbour_bin_y >= n_bins_y)
          {
            continue;
          }

          int neighbour_bin = neighbour_bin_x + n_bins_x * neighbour_bin_y;

          for (unsigned int k = 0; k < verlet_list[neighbour_bin].size(); k++)
          {
            unsigned int neighbour_idx = verlet_list[neighbour_bin][k];
            if (true)
            // if (current_part_idx != neighbour_idx)
            {
              data_type dist = calcualte_distance(pos.row(current_part_idx), pos.row(neighbour_idx));
              if (dist <= c.radius && dist > 0)
              {
                nearIndex[current_part_idx].push_back(neighbour_idx);
                nearDist[current_part_idx].push_back(dist);
              }
            }
          }
        }
      }
    }
  }
}

//   // auto start = std::chrono::high_resolution_clock::now();
//   //     std::vector<std::vector<double>> pos_vec(c.n_particles, std::vector<double>(2, 0));
//   // #pragma omp parallel for num_threads(10)
//   //     for (unsigned int i = 0; i < c.n_particles; i++)
//   //     {
//   //         pos_vec[i][0] = (double)pos(i, 0);
//   //         pos_vec[i][1] = (double)pos(i, 1);
//   //     }
//   std::cout << " pos_vec[0][0] = " << pos_vec[0][0] << std::endl;
//   std::cout << " pos(0,0), pos(0,1) = " << pos(0, 0) << "," << pos(0, 1) << std::endl;
//   std::cout << "  = " << pos(0, 0) << "," << pos(0, 1) << std::endl;
//   KDTree tree(pos_vec);
//   // auto end = std::chrono::high_resolution_clock::now();
//   // std::cerr << "Time for preparing the tree: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

//   // start = std::chrono::high_resolution_clock::now();
//   std::cout << "pos_vec.size() = " << pos_vec.size() << std::endl;

// #pragma omp parallel for num_threads(10)
//   for (unsigned int i = 0; i < pos_vec.size(); i++)
//   {
//     nearIndex[i] = (tree.neighborhood_indices(pos_vec[i], nearDist[i], c.radius));
//   }
//   // end = std::chrono::high_resolution_clock::now();
//   // std::cerr << "Time for finding NN: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1e6 << " seconds\n";
// }
// void print_nodes(const Kdtree::KdNodeVector &nodes)
// {
//   size_t i, j;
//   for (i = 0; i < nodes.size(); ++i)
//   {
//     if (i > 0)
//       std::cout << " ";
//     std::cout << "(";
//     for (j = 0; j < nodes[i].point.size(); j++)
//     {
//       if (j > 0)
//         std::cout << ",";
//       std::cout << nodes[i].point[j];
//     }
//     std::cout << ")";
//   }
//   std::cout << " : ";
// }

// void print_nodes_distance(const Kdtree::KdNodeVector &nodes, std::vector<double> point)
// {
//   size_t i, j;
//   for (i = 0; i < nodes.size(); ++i)
//   {
//     if (i > 0)
//       std::cout << " ";
//     std::cout << "(";
//     for (j = 0; j < nodes[i].point.size(); j++)
//     {
//       if (j > 0)
//         std::cout << ",";
//       std::cout << nodes[i].point[j];
//     }
//     std::cout << ")";
//   }
//   std::cout << " : ";
// }
// void initialise_NN_2(constants &c,
//                      MatrixXX &pos,
//                      std::vector<std::vector<unsigned int>> &nearIndex,
//                      std::vector<std::vector<double>> &nearDist)
// {
//   // auto start = std::chrono::high_resolution_clock::now();
//   Kdtree::KdNodeVector nodes;
//   std::vector<std::vector<double>> pos_vec(c.n_particles, std::vector<double>(2, 0));
// #pragma omp parallel for num_threads(10)
//   for (unsigned int i = 0; i < c.n_particles; i++)
//   {
//     std::vector<data_type> point(2);
//     pos_vec[i][0] = (double)pos(i, 0);
//     pos_vec[i][1] = (double)pos(i, 1);
//     point[0] = pos(i, 0);
//     point[1] = pos(i, 1);
//     nodes.push_back(Kdtree::KdNode(point));
//     // nodes.push_back(Kdtree::KdNode(pos_vec[i]));
//   }
//   // for (int i = 0; i< c.n_particles; i++)
//   // {
//   //     std::vector<data_type> point(2);
//   // point[0] = pos(i, 0);
//   // point[1] = pos(i, 1);
//   // nodes.push_back(Kdtree::KdNode(point));
//   // }

//   Kdtree::KdTree tree(&nodes, 0);
//   std::vector<Kdtree::KdNodeVector> ss;
//   unsigned int count = 0;
//   for (int i = 0; i < c.n_particles; i++)
//   {
//     Kdtree::KdNodeVector nn;
//     std::vector<data_type> point(2);
//     point[0] = pos(i, 0);
//     point[1] = pos(i, 1);
//     tree.range_nearest_neighbors(point, c.radius, &nn);
//     ss.push_back(nn);
//     count = count + nn.size();
//     if (i == 0)
//     {
//       std::cout << "NN for particle 0: ";
//       print_nodes(nn);
//       std::cout << std::endl;
//     }
//   }
//   std::cout << "Total number of neighbors using the new NN when calculated while making: " << count << std::endl;
//   count = 0;
//   for (int i = 0; i < c.n_particles; i++)
//   {
//     count = count + ss[i].size();
//   }
//   std::cout << "Total number of neighbors using the new NN: " << count << std::endl;
// }