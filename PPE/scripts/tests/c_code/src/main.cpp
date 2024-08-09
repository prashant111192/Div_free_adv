#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "type_def.hpp"
#include "NN.hpp"
#include "in_out.hpp"
#include "compute.hpp"
#include "kernel.hpp"


int main()
{
    auto start_complete = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Let's do this!!" << std::endl;
    data_type size = 8;
    data_type dp = 1;
    auto boundary_fac = 2;
    constants c = define_constants(size, dp, boundary_fac);

    print_constants(c);

    MatrixXX pos(c.n_particles, 2);
    MatrixXX vel(c.n_particles, 2);
    MatrixXX density(c.n_particles, 1);
    // std::vector<data_type> p_type(c.n_particles, 1);
    Eigen::MatrixXi p_type(c.n_particles, 1);

    make_particles(c, pos, vel, density, p_type);
    std::cout<< "total number of fluids: "<< p_type.sum() << std::endl;

    std::vector<std::vector<double>> nearDist_(c.n_particles);    // [center particle, neighbor particles] generated from vecDSPH with correspongding idx
    std::vector<std::vector<unsigned>> nearIndex(c.n_particles); // [center particle, neighbor particles] generated from vecDSPH with correspongding idx
    initialise_NN(c, pos, nearIndex, nearDist_);
    std::vector<std::vector<data_type>> nearDist(c.n_particles);
    for (unsigned int i = 0; i < nearDist_.size(); i++)
    {
        nearDist[i].resize(nearDist_[i].size());
        for (unsigned int j = 0; j < nearDist_[i].size(); j++)
        {
            nearDist[i][j] = nearDist_[i][j];
        }
    }
    // std::vector<data_type> nearDist(nearDist_.begin(), nearDist_.end());
    
    // Finding the maximum number of NN
    int count = 0;
    int total_NN = 0;
    int avg_nn = 0;
    for (unsigned int j = 0; j < nearIndex.size(); j++)
    {
        total_NN += nearIndex[j].size();
        if (count < nearIndex[j].size())
        {
            count = nearIndex[j].size();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<< "Sizeof c"<< sizeof(c) << std::endl;
    std::cout<< ">>Time taken for NN: " << duration.count()/1e6 << " seconds\n";
    std::cout << "Maximum number of NN: " << count << std::endl;
    std::cout << "Total number of NN: " << total_NN << std::endl;
    std::cout << "Average number of NN: " << (float)total_NN/nearIndex.size() << std::endl;

    // Creating the GRADIENT and LAPLACIAN Matrix
    start = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<data_type> gradient_x(c.n_particles, c.n_particles);
    std::cout << "Size of gradient_x: " << gradient_x.nonZeros() << std::endl;
    std::cout<< "Size of couit: "<< sizeof(count) << std::endl;
    std::cout<< "Size of gradient_x:"<< sizeof(gradient_x) << std::endl;
    gradient_x.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    std::cout << "Size of gradient_x: " << gradient_x.nonZeros() << std::endl;
    std::cout<< "Size of gradient_x:"<< sizeof(gradient_x) << std::endl;
    Eigen::SparseMatrix<data_type> gradient_y(c.n_particles, c.n_particles);
    gradient_y.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    Eigen::SparseMatrix<data_type> laplacian(c.n_particles, c.n_particles);
    laplacian.reserve(Eigen::VectorXi::Constant(c.n_particles, count));

    prepare_grad_lap_matrix(pos, nearIndex, nearDist, c, gradient_x, gradient_y, laplacian);

    std::cout<< "Size of gradient_x:"<< sizeof(gradient_x) << std::endl;
    std::cout << "Size of gradient_x: " << gradient_x.nonZeros() << std::endl;
    // data_type just_check = gradient_x.cwiseProduct(gradient_x).sum();
    // int numberOfNonZeroElements = gradient_x.nonZeros();
    // std::cout << "Number of non-zero elements in _X: " << numberOfNonZeroElements << std::endl;
    data_type just_check = gradient_x.sum();
    std::cout<< "Result of sum(gradient_x): " << just_check << std::endl;
    // just_check = gradient_y.cwiseProduct(gradient_y).sum();
    // just_check = gradient_y.sum();
    // just_check = (gradient_y*gradient_y).sum();
    // numberOfNonZeroElements = gradient_y.nonZeros();
    // std::cout << "Number of non-zero elements in _y: " << numberOfNonZeroElements << std::endl;
    // std::cout<< "Result of gradient_y*gradient_y: " << just_check << std::endl;
    // just_check = laplacian.cwiseProduct(laplacian).sum();
    // just_check = laplacian.sum();
    // just_check = (laplacian*laplacian).sum();
    // numberOfNonZeroElements = laplacian.nonZeros();
    // std::cout << "Number of non-zero elements in lap: " << numberOfNonZeroElements << std::endl;
    // std::cout<< "Result of laplacian: " << just_check << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << ">>Time taken to compute the gradient and laplacian matrix : "
              << duration.count() / 1e6 << " seconds" << std::endl;

    // // DIVERGENCE
    start = std::chrono::high_resolution_clock::now();
    MatrixXX divergence(c.n_particles, 1);
    divergence.fill(0);
    calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);
    // std::cout<< "max divergence: " << divergence.maxCoeff() << "\n";
    writeMatrixToFile(divergence, "divergence.csv");
    // writeMatrixToFile(pos, "pos.csv");
    // writeMatrixToFile(vel, "vel.csv");
    // writeMatrixToFile(p_type, "p_type.csv");
    // exit(0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<< ">>Time taken for divergence: " << duration.count()/1e6 << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    pressure_poisson(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, laplacian, c);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<< ">>Time taken for pressure_poisson: " << duration.count()/1e6 << " seconds\n";

    writeMatrixToFile(p_type, "p_type.csv");
    writeMatrixToFile(divergence, "divergence.csv");
    writeMatrixToFile(pos, "pos.csv");
    writeMatrixToFile(vel, "vel.csv");
    // std::cout << "resolution: " << c.resolution << std::endl;
    // std::cout << "n_particles: " << c.n_particles << std::endl;
    // size_t total_NN;
    // size_t NN_i= 0;
    // for (unsigned int i = 0; i < c.n_particles; i++)
    // {
    //     NN_i += nearIndex[i].capacity();
    // }
    // total_NN = NN_i * sizeof(unsigned int);
    // std::cout << "Total memory used for NN_index: " << total_NN/(1024*1024) << " Mbytes" << std::endl;
    // total_NN = NN_i * sizeof(data_type);
    // std::cout << "Total memory used for NN_distance: " << total_NN/(1024*1024) << " Mbytes" << std::endl;
    // std::cout<< "size of double: " << sizeof(data_type) << "and size of unsigned int: " << sizeof(unsigned int) << std::endl;
    divergence = divergence.array().abs();
    std::cout << "Max Divergence: " << divergence.maxCoeff() << std::endl;
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_complete);
    std::cout << ">>Time taken for the complete code: " << duration.count() / 1e6 << " seconds" << std::endl;

    return 0;
}