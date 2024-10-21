#include "compute.hpp"

void calc_divergence(const MatrixXX &pos,
                     const MatrixXX &vel,
                     const MatrixXX &density,
                     const Eigen::MatrixXi &p_type,
                     const std::vector<std::vector<unsigned int>> &nearIndex,
                     const std::vector<std::vector<data_type>> &nearDist,
                     MatrixXX &divergence,
                     const Eigen::SparseMatrix<data_type> &gradient_x,
                     const Eigen::SparseMatrix<data_type> &gradient_y,
                     const constants &c)
{
    LOG(INFO) << "Calculating the divergence";
    auto start = std::chrono::high_resolution_clock::now();
    divergence.fill(0);
    int count_fluid = 0;
    data_type total_abs_div = 0;
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        if (p_type(i) == 1) // if Fluid particle
        {
            count_fluid += 1;
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                // if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                // //    if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && p_type[nearIndex[i][j]] == 1)
                // {
                MatrixXX r_ij(1, 2);
                MatrixXX weight(1, 2);
                weight.fill(0);
                MatrixXX v_ji(1, 2);

                r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                v_ji = vel.row(nearIndex[i][j]) - vel.row(i);
                weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);

                data_type temp = weight.row(0).dot(v_ji.row(0));

                temp = temp * c.mass * nearDist[i][j] / (nearDist[i][j]);
                divergence(i) = divergence(i) + temp / density(i);
                // }
            }
        }
        total_abs_div += abs(divergence(i));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Time taken to calculate the divergence: " << duration.count() / 1e6 << " seconds";
}

void pressure_poisson(const MatrixXX &pos,
                      MatrixXX &vel,
                      const MatrixXX &density,
                      const Eigen::MatrixXi &p_type,
                      const std::vector<std::vector<unsigned int>> &nearIndex,
                      const std::vector<std::vector<data_type>> &nearDist,
                      MatrixXX &divergence,
                      const Eigen::SparseMatrix<data_type> &gradient_x,
                      const Eigen::SparseMatrix<data_type> &gradient_y,
                      const Eigen::SparseMatrix<data_type> &laplacian,
                      const MatrixXX &normals,
                      const constants &c,
                      const unsigned int count)
{
    LOG(INFO) << "Starting the pressure poisson solver";
    int max_iter = 100;
    int current_iter = 0;
    SpMatrixXX A(c.n_particles, c.n_particles);
    MatrixXX b(c.n_particles, 1);
    MatrixXX p(c.n_particles, 1);
    MatrixXX pos_write(c.n_particles, 2);
    MatrixXX vel_write(c.n_particles, 2);
    MatrixXX div_write(c.n_particles, 1);

    // std::thread th_write1, th_write2;

    data_type max_div = 1000;
    std::cout << "#Run;#Iter;Error;MaxDiv" << std::endl;
    while (max_div > 1e-6 && current_iter < max_iter)
    {
        current_iter++;

        A.setZero();
        b.setZero();
        p.setZero(); // solution matrix
        A.reserve(Eigen::VectorXi::Constant(c.n_particles, count));
        auto start = std::chrono::high_resolution_clock::now();

        LOG(INFO) << "Preparing the A matrix";
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i < c.n_particles; i++)
        {
            if (p_type(i) == 3) // For solid particles
            {
                // continue; // skip the solid particles IMPORTANT!!!!!!
                for (unsigned int j = 0; j < nearIndex[i].size(); j++)
                {
                    data_type a_ij;
                    a_ij = c.mass / density(i);
                    MatrixXX grad_mat(1, 2);
                    grad_mat(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    grad_mat(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                    a_ij = a_ij * (grad_mat.row(0).dot(normals.row(i)));
#pragma omp critical(foo1)
                    A.insert(i, nearIndex[i][j]) = a_ij;
                }
            }

            else
            {
                for (unsigned int j = 0; j < nearIndex[i].size(); j++)
                {
                    if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                    {
                        MatrixXX v_ji(1, 2);
                        MatrixXX grad_mat(1, 2);
                        data_type temp;
                        v_ji = vel.row(nearIndex[i][j]) - vel.row(i);

                        data_type lap = laplacian.coeff(i, nearIndex[i][j]);

#pragma omp critical(foo2)
                        A.insert(i, nearIndex[i][j]) = lap * c.mass / density(i);

                        grad_mat(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                        grad_mat(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                        temp = v_ji.row(0).dot(grad_mat.row(0));
                        b(i) = b(i) + temp * c.mass / density(i);
                    }
                }
            }
        }
#pragma omp barrier

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        LOG(INFO) << "Done with patrtial creation of A and it took " << duration.count() / 1e6 << " seconds";
        start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i < c.n_particles; i++)
        {
            data_type sum = 0;
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (j == i) continue;

                sum = sum - A.coeff(i, nearIndex[i][j]);
            }
#pragma omp critical(foo)
            A.insert(i, i) = sum;
        }
#pragma omp barrier
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        LOG(INFO) << "Done with the diagonal elements of A and it took " << duration.count() / 1e6 << " seconds";

        // Sanity check
        // just to count the number of negative diagonal elements
        start = std::chrono::high_resolution_clock::now();
        unsigned int count_negative_diagonal = 0;
        unsigned int count_zero_diagonal = 0;
        for (unsigned int i = 0; i < c.n_particles; i++)
        {
            if (A.coeff(i, i) < 0)
            {
                count_negative_diagonal++;
            }
            if (A.coeff(i, i) == 0)
            {
                count_zero_diagonal++;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        LOG(INFO) << "Done with the sanity check and it took " << duration.count() / 1e6 << " seconds";

        using namespace Eigen;

        // LEAST SQUARES Conjugate Gradient SOLVER
        LeastSquaresConjugateGradient<SparseMatrix<data_type>> lscg;
        DiagonalPreconditioner<data_type> precond;
        // lscg.setMaxIterations(500);
        lscg.setTolerance(1e-12);
        lscg.compute(A);
        p = lscg.solve(b);

        MatrixXX q(c.n_particles, 2);
        q = cal_div_part_vel(pos, density, p_type, nearIndex, nearDist, p, gradient_x, gradient_y, c);
        vel = vel - q;
        calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);

        max_div = divergence.maxCoeff();
        // if (current_iter % 1 == 0)
        std::cout << current_iter << ";" << lscg.iterations() << ";" << lscg.error() << ";" << max_div << std::endl;
        int write_freq = 2;
        if (current_iter % write_freq == 0)
        {
            // pos_write = pos;
            // div_write = divergence;
            // vel_write = vel;
            std::string filename = std::to_string(c.dp_i) + "_divergence_" + std::to_string(current_iter) + ".csv";
            // th_write1 = std::thread(&writeMatrixToFile<MatrixXX&>, pos_write, div_write, filename);
            writeMatrixToFile<MatrixXX &>(pos, divergence, filename);
            filename = std::to_string(c.dp_i) + "_velocity_" + std::to_string(current_iter) + ".csv";
            // th_write2 = std::thread(&writeMatrixToFile<MatrixXX&>, pos_write, vel_write, filename);
            writeMatrixToFile<MatrixXX>(pos, vel, filename);
        }
        // if (current_iter % write_freq == write_freq-1 || current_iter == max_iter-1)

        // {
        //     th_write1.join();
        //     th_write2.join();
        // }

        // if (max_div < 1e-5)
        // {
        //     break;
        // }
    }
    // if (current_iter == max_iter - 1)
    // {
    //     std::cerr << "Pressure Poisson Solver did not converge with the max divergence being " << max_div << std::endl;
    // }
    // else
    // {
    //     std::cerr << "Pressure Poisson Solver converged at " << current_iter << " with the max divergence being " << max_div << std::endl;
    // }
}

MatrixXX cal_div_part_vel(const MatrixXX &pos,
                          const MatrixXX &density,
                          const Eigen::MatrixXi &p_type,
                          const std::vector<std::vector<unsigned int>> &nearIndex,
                          const std::vector<std::vector<data_type>> &nearDist,
                          const MatrixXX &p,
                          const Eigen::SparseMatrix<data_type> &gradient_x,
                          const Eigen::SparseMatrix<data_type> &gradient_y,
                          const constants &c)
{
    LOG(INFO) << "Calculating the divergence part of the velocity";
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXX q(c.n_particles, 2);
    q.fill(0);
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        if (p_type(i) == 1) // For fluid particles
        {
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                {

                    MatrixXX weight(1, 2);
                    weight.fill(0);
                    MatrixXX temp_mat(1, 2);
                    data_type temp;

                    weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                    temp = p(nearIndex[i][j]) - p(i);
                    weight = weight * temp;
                    temp_mat = weight * c.mass / density(nearIndex[i][j]);
                    q.row(i) = q.row(i) + temp_mat; // these are FUCKING VECTORS!!! dont forget
                }
            }
        }
        else // for solids
        {
            continue;
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                {

                    MatrixXX r_ij(1, 2);
                    MatrixXX weight(1, 2);
                    weight.fill(0);
                    MatrixXX temp_mat(1, 2);
                    data_type temp;

                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                    temp = p(nearIndex[i][j]) - p(i);
                    weight = weight * temp;
                    temp_mat = weight * c.mass / density(nearIndex[i][j]);
                    q.row(i) = q.row(i) + temp_mat; // these are FUCKING VECTORS!!! dont forget
                }
            }
        }
    }
#pragma omp barrier
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Done with calculation of the divergent part of the velocity and it took "<< duration.count()/1e6<<" seconds.";
    return q;
}

void saving_grads(const MatrixXX &pos,
                    MatrixXX &vel,
                    const MatrixXX &density,
                    const Eigen::MatrixXi &p_type,
                    const std::vector<std::vector<unsigned int>> &nearIndex,
                    const std::vector<std::vector<data_type>> &nearDist,
                    MatrixXX &divergence,
                    const Eigen::SparseMatrix<data_type> &gradient_x,
                    const Eigen::SparseMatrix<data_type> &gradient_y,
                    const Eigen::SparseMatrix<data_type> &laplacian,
                    const MatrixXX &normals,
                    const constants &c,
                    const unsigned int count)
{
    

}