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
  LOG(INFO) << "Initialising the NN";
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
  LOG(INFO) << "Total number of bins: " << total_bins;

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
