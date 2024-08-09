#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

struct Particle
{
    std::vector<double> pos;
    unsigned idx;
    // Particle() : pos{0,0}, idx(0) {}
    Particle(double x, double y, unsigned idx) : pos{x, y}, idx(idx) {}
};

void Print_Verlet_List(std::vector<std::vector<unsigned int>> verlet_list)
{
    for (unsigned int i = 0; i < verlet_list.size(); i++)
    {
        std::cout << "Bin " << i << " contains particles: ";
        for (unsigned int j = 0; j < verlet_list[i].size(); j++)
        {
            std::cout << verlet_list[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

double Distance(Particle &p1, Particle &p2)
{
    double dist = 0;
    for (unsigned int i = 0; i < p1.pos.size(); i++)
    {
        dist += (p1.pos[i] - p2.pos[i]) * (p1.pos[i] - p2.pos[i]);
    }
    dist = std::sqrt(dist);
    return dist;
}

void Print_nn_list(std::vector<std::vector<unsigned int>> nn_list, std::vector<Particle> points)
{
    // Printing out the neighbour list and the distance between the neighbouring particles
    unsigned count = 0;
    for (unsigned int i = 0; i < nn_list.size(); i++)
    {
        std::cout << "Particle " << i << " has neighbours: ";
        for (unsigned int j = 0; j < nn_list[i].size(); j++)
        {
            std::cout << nn_list[i][j] << " ";
            std::cout << "Distance between the particles: " << Distance(points[i], points[nn_list[i][j]]) << std::endl;
            count++;
        }
    }
    std::cout << "Total number of neighbours: " << count << std::endl;
}

int main()
{
    unsigned int n = 100;
    // Define the points
    std::vector<Particle> points;
    for (unsigned int i = 0; i < 10; i++)
    {
        for (unsigned int j = 0; j < 10; j++)
        {
            // int ii  = 1 + (int) (10.0 * (rand() / (RAND_MAX + 1.0)));
            // int jj  = 1 + (int) (10.0 * (rand() / (RAND_MAX + 1.0)));
            points.push_back(Particle(i, j, i * 10 + j));
        }
    }

    double bin_size = 5;
    double min_x = -1 - bin_size;
    double max_x = 11 + bin_size;
    double min_y = -1 - bin_size;
    double max_y = 11 + bin_size;

    int n_bins_x = (max_x - min_x) / bin_size;
    int n_bins_y = (max_x - min_x) / bin_size;
    int total_bins = n_bins_x * n_bins_x;

    // Define the domain --> 0-10
    double radius = 3;

    std::vector<std::vector<unsigned int>> nn_list(n);
    std::vector<std::vector<unsigned int>> verlet_list(total_bins);

    for (unsigned int i = 0; i < n; i++)
    {
        int x = points[i].pos[0] - min_x;
        int y = points[i].pos[1] - min_y;

        unsigned bin_x = x / bin_size;
        unsigned bin_y = y / bin_size;
        unsigned bin = bin_x + n_bins_x * (bin_y);
        // unsigned int xidx = x / bin_size;
        // xidx = xidx >= n_bins_x ? n_bins_x : xidx;
        verlet_list[bin].push_back(points[i].idx);
    }

    int dim = 2;
        std::vector<std::vector<unsigned>> verlet_neighbours(n);

    for (unsigned int i = 0; i < total_bins; i++) {
        for (unsigned int j = 0; j < verlet_list[i].size(); j++) {
            unsigned int particle_idx = verlet_list[i][j];
            Particle& p1 = points[particle_idx];

            for (int x_itr = -1; x_itr <= 1; x_itr++) {
                for (int y_itr = -1; y_itr <= 1; y_itr++) {
                    int neighbour_bin_x = (i % n_bins_x) + x_itr;
                    int neighbour_bin_y = (i / n_bins_x) + y_itr;

                    if (neighbour_bin_x < 0 || neighbour_bin_x >= n_bins_x ||
                        neighbour_bin_y < 0 || neighbour_bin_y >= n_bins_y) {
                        continue;
                    }

                    int neighbour_bin = neighbour_bin_x + n_bins_x * neighbour_bin_y;

                    for (unsigned int k = 0; k < verlet_list[neighbour_bin].size(); k++) {
                        unsigned int neighbour_idx = verlet_list[neighbour_bin][k];
                        Particle& p2 = points[neighbour_idx];

                        if (particle_idx != neighbour_idx && Distance(p1, p2) <= radius) {
                            verlet_neighbours[particle_idx].push_back(neighbour_idx);
                        }
                    }
                }
            }
        }
    }

    Print_Verlet_List(verlet_list);
    Print_nn_list(verlet_neighbours, points);

    return 0;
}
