#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "type_def.hpp"
#include "NN.hpp"
#include "in_out.hpp"
#include "compute.hpp"
#include "kernel.hpp"
#include "easylogging++.cc"
#include "log.hpp"


INITIALIZE_EASYLOGGINGPP

void start(int dp_i);


int main(int argc, char* argv[])
{
    // Stting up the logger
    START_EASYLOGGINGPP(argc, argv);
    el::Configurations conf("./../../src/log_easyconfig.conf");
    el::Loggers::reconfigureAllLoggers(conf);
    el::Logger* DATALogger = el::Loggers::getLogger("DATA");
    el::Configurations conf2("./../../src/log_data_config.conf");
    el::Loggers::reconfigureLogger(DATALogger, conf2);
    el::Logger* TIMELogger = el::Loggers::getLogger("TIME");
    el::Configurations conf3("./../../src/log_time_config.conf");
    el::Loggers::reconfigureLogger(TIMELogger, conf3);

    LOG(INFO) << "Starting the simulation with different dp_i (factor to scale the radius of influence)"; 
    for (int i = 10; i <= 100; i++)
    {
        LOG(INFO)<< "Starting simualtion with Factor for radius of influence: " << i;
        start(i);
        LOG(INFO) << "==================================";
        LOG(INFO) << "==================================\n";
    }
    return 0;
}

/**
 * Initializes and simulates the particle system.
 *
 * This function sets up the simulation by initializing particle properties such as position,
 * velocity, density, and type. It configures the neighborhood data structures for particle
 * interaction and prepares matrices for gradient and Laplacian calculations. The function
 * also calculates divergence and executes the pressure Poisson solver to update particle
 * velocities. Logging statements are used to track the progress and performance of each step.
 *
 * @param dp_i An integer representing the scaling factor for the radius of influence.
 */
void start(int dp_i)
{
    auto start_complete = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    data_type size = 1;
    data_type dp = 0.01;
    data_type boundary_fac = 20*dp;

    constants c = define_constants(size, dp, boundary_fac, dp_i);
    LOG(INFO) << c << std::endl;

    LOG(INFO) << "Intialising particle arrays";
    MatrixXX pos(c.n_particles, 2);
    std::cout<< "size: " << pos.rows() << std::endl;
    pos.fill(0);
    MatrixXX vel(c.n_particles, 2);
    vel.fill(0);
    MatrixXX density(c.n_particles, 1);
    density.fill(1000);
    Eigen::MatrixXi p_type(c.n_particles, 1); // 1 = fluid, 0 = boundary
    MatrixXX pressure(c.n_particles, 1);

    // make_particles(c, pos, vel, density, p_type);
    make_from_dsph(c, pos, vel, density, p_type, pressure);
    MatrixXX normals_computed(c.n_particles, 2);  // Normals can be computed only after NN
    normals_computed.fill(0);

    // writeMatrixToBinaryFile<MatrixXX&>(pos, vel, std::to_string(dp_i)+"vel_ini");
    // writeMatrixToFile<MatrixXX&>(pos, vel, std::to_string(dp_i)+"vel_ini");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    CLOG(INFO, "TIME") << "Initialise particles(s): " << duration.count()/1e6;

    LOG(INFO) << "Setting up the NN";
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<data_type>> nearDist(c.n_particles);
    std::vector<std::vector<unsigned>> nearIndex(c.n_particles); // [center particle, neighbor particles] generated from vecDSPH with correspongding idx
    initialise_NN(c, pos, nearIndex, nearDist);
    // Finding the maximum number of NN
    unsigned int count = 0;
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
    LOG(INFO) << "Maximum number of NN: " << count;
    LOG(INFO) << "Avergae number of NN: " << (float)total_NN/nearIndex.size();
    LOG(INFO) << "Maximum number of NN: " << count;
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    CLOG(INFO, "TIME")<< "Find NN(s): " << duration.count()/1e6;

    SpMatrixXX gradient_x(c.n_particles, c.n_particles);
    gradient_x.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    gradient_x.setZero();
    SpMatrixXX gradient_y(c.n_particles, c.n_particles);
    gradient_y.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    gradient_y.setZero();
    SpMatrixXX laplacian(c.n_particles, c.n_particles);
    laplacian.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    laplacian.setZero();

    //     std::cout<< "IN MAIN"<< std::endl;
    // for (unsigned int i = 0; i < c.n_particles; i++){
    //     if (pos(i, 0) > -0.1 && pos(i, 0) < 0.1 && pos(i, 1) > -0.1 && pos(i, 1) < 0.1)
    //     {
    //         std::cout<< "position:" << pos(i, 0) << " " << pos(i, 1) << std::endl;
    //         std::cout<< "density:" << density(i) << std::endl;
    //         std::cout<< "type:" << p_type(i) << std::endl;
    //     }
    // }

    prepare_grad_lap_matrix_fast(pos, nearIndex, nearDist, c, gradient_x, gradient_y, laplacian, count);
    // prepare_grad_lap_matrix_fast(pos, nearIndex, nearDist, c, gradient_x, gradient_y, laplacian);
    make_normals(c, pos, normals_computed, gradient_x, gradient_y, p_type, nearIndex, density);
    writeMatrixToFile<MatrixXX&>(pos, normals_computed, std::to_string(dp_i)+"normals_computed");
    // writeMatrixToFile<Eigen::MatrixXi&>(pos, p_type, std::to_string(dp_i)+"particle_type");
    //     std::cout<< "IN MAIN"<< std::endl;
    // for (unsigned int i = 0; i < c.n_particles; i++){
    //     if (pos(i, 0) > -0.1 && pos(i, 0) < 0.1 && pos(i, 1) > -0.1 && pos(i, 1) < 0.1)
    //     {
    //         std::cout<< "position:" << pos(i, 0) << " " << pos(i, 1) << std::endl;
    //         std::cout<< "density:" << density(i) << std::endl;
    //         std::cout<< "type:" << p_type(i) << std::endl;
    //     }
    // }


    // DIVERGENCE
    MatrixXX divergence(c.n_particles, 1);
    divergence.fill(0);
    calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);
    std::string filename = std::to_string(dp_i)+"_divergence";
    writeMatrixToFile<MatrixXX&>(pos, divergence, filename);
    divergence = divergence.array().abs();
    CLOG(INFO, "DATA")  <<dp_i * 0.2<< ";"<< divergence.maxCoeff() ;
/*
    pressure_poisson(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, laplacian, normals_computed, c, count);

    writeMatrixToFile<Eigen::MatrixXi&>(pos, p_type, std::to_string(dp_i)+"_p_type");
    writeMatrixToFile<MatrixXX&>(pos, divergence, std::to_string(dp_i)+"divergence_2");
    writeMatrixToFile<MatrixXX&>(pos, vel, std::to_string(dp_i)+"vel2");
    // divergence = divergence.array().abs();
    // CLOG(INFO, "DATA")  <<dp_i<< ";"<< divergence.maxCoeff() ;
    */
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_complete);
    CLOG(INFO, "TIME") << "Total time taken for the simulation(s): " << duration.count()/1e6;
}

