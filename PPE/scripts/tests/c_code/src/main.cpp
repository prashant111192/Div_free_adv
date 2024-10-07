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
    el::Configurations conf("./../../src/easyconfig.conf");
    el::Loggers::reconfigureAllLoggers(conf);
    // el::Logger* DATALogger = el::Loggers::getLogger("DATA");
    // el::Configurations conf2("./../../src/config_data.conf");
    // el::Loggers::reconfigureLogger(DATALogger, conf2);


    // conf.set(el::Level::Global, el::ConfigurationType::Format, "%datetime %level %msg");
    // conf.set(el::Level::Global, el::ConfigurationType::Filename, "logs/my_log.log");
    // el::Loggers::reconfigureLogger("default", conf);

    // Step 3: Use custom logging levels
    
    // Use CLOG macro for logging with custom levels

    LOG(INFO) << "Starting the simulation with different dp_i (factor to scale the radius of influence)"; 
    for (int i = 2; i <= 12; i=i+2)
    {
        LOG(INFO)<< "Starting simualtion with dp_i: " << i;
        start(i);
    }
    return 0;
}

void start(int dp_i)
{
    auto start_complete = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    data_type size = 100;
    data_type dp = 1;
    auto boundary_fac = 20*dp;

    constants c = define_constants(size, dp, boundary_fac, dp_i);
    LOG(INFO) << c;

    LOG(INFO) << "Intialising particle arrays";
    MatrixXX pos(c.n_particles, 2);
    pos.fill(0);
    MatrixXX vel(c.n_particles, 2);
    vel.fill(0);
    MatrixXX density(c.n_particles, 1);
    density.fill(1000);
    Eigen::MatrixXi p_type(c.n_particles, 1);
    MatrixXX normals(c.n_particles, 2);
    normals.fill(0);
    make_particles(c, pos, vel, density, p_type, normals);

    writeMatrixToFile<MatrixXX&>(pos, vel, std::to_string(dp_i)+"vel_ini.csv");
    writeMatrixToFile<MatrixXX&>(pos, normals, std::to_string(dp_i)+"normals.csv");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Time taken to initialise the particles: " << duration.count()/1e6 << " seconds";

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
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO)<< "Time taken to initialise the NN: " << duration.count()/1e6 << " seconds";

    SpMatrixXX gradient_x(c.n_particles, c.n_particles);
    gradient_x.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    gradient_x.setZero();
    SpMatrixXX gradient_y(c.n_particles, c.n_particles);
    gradient_y.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    gradient_y.setZero();
    SpMatrixXX laplacian(c.n_particles, c.n_particles);
    laplacian.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
    laplacian.setZero();
    data_type sum_temp = laplacian.sum();

    prepare_grad_lap_matrix(pos, nearIndex, nearDist, c, gradient_x, gradient_y, laplacian);

    // DIVERGENCE
    MatrixXX divergence(c.n_particles, 1);
    divergence.fill(0);
    calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);
    std::string filename = std::to_string(dp_i)+"_divergence.csv";
    writeMatrixToFile<MatrixXX&>(pos, divergence, filename);

    pressure_poisson(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, laplacian, normals, c, count);

    writeMatrixToFile<Eigen::MatrixXi&>(pos, p_type, std::to_string(dp_i)+"_p_type.csv");
    writeMatrixToFile<MatrixXX&>(pos, divergence, std::to_string(dp_i)+"divergence_2.csv");
    writeMatrixToFile<MatrixXX&>(pos, vel, std::to_string(dp_i)+"vel2.csv");
    divergence = divergence.array().abs();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_complete);
    LOG(INFO) << "Total time taken for the simulation: " << duration.count()/1e6 << " seconds";
}