#include <iostream>
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
                if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                //    if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && p_type[nearIndex[i][j]] == 1)
                {
                    MatrixXX r_ij(1, 2);
                    MatrixXX weight(1, 2);
                    weight.fill(0);
                    MatrixXX v_ji(1, 2);

                    // std::cout<< "pos.row(i): "<< pos.row(i) << std::endl;
                    // std::cout<< "pos.row(nearIndex[i][j]): "<< pos.row(nearIndex[i][j]) << std::endl;
                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    // std::cout<< "r_ij: "<< r_ij << std::endl;
                    v_ji = vel.row(nearIndex[i][j]) - vel.row(i);
                    // std::cout<< "v_ji: "<< v_ji << std::endl;
                    // weight = gradient_poly6(nearDist[i][j], c, r_ij);
                    weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);

                    data_type temp = weight.row(0).dot(v_ji.row(0));
                    if (temp != 0)
                    {
                        // std::cout<< "weight: "<< weight << std::endl;
                        // std::cout<< "v_ji: "<< v_ji << std::endl;
                        // std::cout<< "dot product: "<< temp << std::endl;
                    }
                    // std::cout<< pos.row(i) << "\t" << pos.row(nearIndex[i][j]) <<"\t" << r_ij<< "weight: " << weight
                    // << "\ttemp: "<<temp<<  std::endl;

                    // temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]+c.Eta);
                    temp = temp * c.mass * nearDist[i][j] / (nearDist[i][j]);
                    divergence(i) = divergence(i) + temp / density(i);
                }
            }
        }
        total_abs_div += abs(divergence(i));
    }
    std::cout << "Total absolute divergence: " << total_abs_div << std::endl;
    std::cout << "Number of fluid particles: " << count_fluid << std::endl;
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
                      const constants &c)
{
    std::cout << "Starting the pressure poisson solver" << std::endl;
    int max_iter = 100000;
    int current_iter = 0;
    SpMatrixXX A(c.n_particles, c.n_particles);
    MatrixXX b(c.n_particles, 1);
    MatrixXX p(c.n_particles, 1);

    data_type max_div = 1000;
    while (max_div > 1e-6 && current_iter < max_iter)
    {
        current_iter++;

        A.setZero();
        b.setZero();
        p.setZero(); // solution matrix
        A.reserve(Eigen::VectorXi::Constant(c.n_particles, 600));
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Starting the creation of A" << std::endl;
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i < c.n_particles; i++)
        {
            if (p_type(i) == 3) // For solid particles
            {
                // continue; // skip the solid particles IMPORTANT!!!!!!
                for (unsigned int j = 0; j < nearIndex[i].size(); j++)
                {
                    if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                    {
                        data_type a_ij;
                        a_ij = c.mass / density(i);
                        MatrixXX grad_mat(1, 2);
                        grad_mat(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                        grad_mat(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                        a_ij = a_ij * (grad_mat.row(0).dot(normals.row(i)));
                        // A.coeffRef(i, nearIndex[i][j]) = a_ij;
                        A.insert(i, nearIndex[i][j]) = a_ij;
                    }
                }
            }

            else
            {
                for (unsigned int j = 0; j < nearIndex[i].size(); j++)
                {
                    if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                    {
                        // MatrixXX r_ij(1, 2);
                        MatrixXX v_ji(1, 2);
                        MatrixXX grad_mat(1, 2);
                        data_type temp;
                        // r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                        v_ji = vel.row(nearIndex[i][j]) - vel.row(i);

                        // data_type lap = lap_poly6(nearDist[i][j], c);
                        data_type lap = laplacian.coeff(i, nearIndex[i][j]);

                        // A.coeffRef(i, nearIndex[i][j]) = lap*c.mass/density(i);
                        A.insert(i, nearIndex[i][j]) = lap * c.mass / density(i);
                        // grad_mat = gradient_poly6(nearDist[i][j], c, r_ij);
                        grad_mat(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                        grad_mat(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                        temp = v_ji.row(0).dot(grad_mat.row(0));
                        b(i) = b(i) + temp * c.mass / density(i);
                    }
                }
            }
        }
#pragma omp barrier
        std::cout << "Done with patrtial creation of A" << std::endl;
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i < c.n_particles; i++)
        {

            data_type sum = 0;
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (j == i)
                    continue;

                sum = sum - A.coeff(i, nearIndex[i][j]);
            }
            A.insert(i, i) = sum;
        }
#pragma omp barrier
        // std::cout<< A.nonZeros()<< std::endl;
        std::cout << "created the diagonal elemetes\n";

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cerr << ">>Time taken to prepare part of A: " << duration.count() / 1e6 << " seconds\n";

        // Sanity check
        // just to count the number of negative diagonal elements
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
        std::cout << "Number of negative diagonal elements: " << count_negative_diagonal << std::endl;
        std::cout << "Number of zero diagonal elements: " << count_zero_diagonal << std::endl;

        using namespace Eigen;

        // LEAST SQUARES Conjugate Gradient SOLVER
        LeastSquaresConjugateGradient<SparseMatrix<data_type>> lscg;
        DiagonalPreconditioner<data_type> precond;
        // lscg.setMaxIterations(500);
        lscg.setTolerance(1e-12);
        lscg.compute(A);
        p = lscg.solve(b);
        // std::cout << " Estimated Error: " << lscg.error() << std::endl;
        // std::cout << "Number of iterations: " << lscg.iterations() << std::endl;

        MatrixXX q(c.n_particles, 2);
        q = cal_div_part_vel(pos, density, p_type, nearIndex, nearDist, p, gradient_x, gradient_y, c);
        vel = vel - q;
        calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);

        max_div = divergence.maxCoeff();
        if (current_iter % 1 == 0)
        {
            std::cerr << "Current iteration: " << current_iter << " with the max divergence being " << max_div << std::endl;
            // std::cerr << "Time taken to solve the system(s);" << duration.count() / 1e6 << ";";
            std::cout << "#iterations for lscg;" << lscg.iterations() << ";Error;" << lscg.error() << std::endl;
        }
        if (current_iter % 10 == 0)
        {
            std::string filename = "divergence_" + std::to_string(current_iter) + ".csv";
            writeMatrixToFile(pos, divergence, filename);
            filename = "velocity_" + std::to_string(current_iter) + ".csv";
            writeMatrixToFile(pos, vel, filename);
        }

        if (max_div < 1e-5)
        {
            break;
        }
    }
    if (current_iter == max_iter - 1)
    {
        std::cerr << "Pressure Poisson Solver did not converge with the max divergence being " << max_div << std::endl;
    }
    else
    {
        std::cerr << "Pressure Poisson Solver converged at " << current_iter << " with the max divergence being " << max_div << std::endl;
    }
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
    MatrixXX q(c.n_particles, 2);
    q.fill(0);
// std::cout<< "Here1"<<std::endl;
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        if (p_type(i) == 1) // For fluid particles
        {
            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                {

                    // MatrixXX r_ij(1,2);
                    MatrixXX weight(1, 2);
                    weight.fill(0);
                    MatrixXX temp_mat(1, 2);
                    data_type temp;

                    // r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
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
                    // weight = gradient_poly6(nearDist[i][j], c, r_ij);
                    temp = p(nearIndex[i][j]) - p(i);
                    // temp_mat = p.row(nearIndex[i][j]) - p.row(i);
                    // weight = weight.cwiseProduct(temp_mat);
                    // weight = weight.row(0).dot(temp_mat.row(0));
                    weight = weight * temp;
                    temp_mat = weight * c.mass / density(nearIndex[i][j]);
                    // std::cout<< "temp_mat: "<< temp_mat << std::endl;
                    q.row(i) = q.row(i) + temp_mat; // these are FUCKING VECTORS!!! dont forget
                    // std::cout<< "q: "<< q<< std::endl;
                }
            }
        }
    }
#pragma omp barrier
    // std::cerr<<"Here6"<< std::endl;
    return q;
}
