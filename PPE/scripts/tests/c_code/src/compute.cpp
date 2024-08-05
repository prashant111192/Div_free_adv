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
#pragma omp parallel for num_threads(10)
    int count_fluid = 0;
    data_type total_abs_div = 0;
    for(unsigned int i = 0; i<c.n_particles; i++)
    {
        if (p_type(i) == 1) // if Fluid particle
        {
            count_fluid += 1;
            for(unsigned int j=0; j<nearIndex[i].size(); j++)
            {
               if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
            //    if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && p_type[nearIndex[i][j]] == 1)
                {
                    MatrixXX r_ij(1,2);
                    MatrixXX weight(1,2);
                    weight.fill(0);
                    MatrixXX v_ji(1,2);

                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    v_ji = vel.row(nearIndex[i][j]) - vel.row(i);
                    // weight = gradient_poly6(nearDist[i][j], c, r_ij);
                    weight(0,0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0,1) = gradient_y.coeff(i, nearIndex[i][j]);

                    data_type temp = weight.row(0).dot(v_ji.row(0));
                    // std::cout<< pos.row(i) << "\t" << pos.row(nearIndex[i][j]) <<"\t" << r_ij<< "weight: " << weight 
                                // << "\ttemp: "<<temp<<  std::endl;

                    // temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]+c.Eta);
                    temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]);
                    divergence(i) = divergence(i)+ temp/density(i);
                }
                
            }

        }
        total_abs_div += abs(divergence(i));
    }
    std::cout<< "Total absolute divergence: "<< total_abs_div<< std::endl;
    std::cout<< "Number of fluid particles: "<< count_fluid<< std::endl;
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
                const constants &c)
{
    int max_iter = 100;
    int current_iter = 0;
    SpMatrixXX A(c.n_particles, c.n_particles);
    // A.reserve(VectorXi::Constant(600,c.n_particles));
    MatrixXX b(c.n_particles, 1);
    MatrixXX p(c.n_particles, 1);
    // VectorX b(c.n_particles);
    // VectorX p(c.n_particles);


    // std::cout<<"Here-1"<<std::endl;
    data_type max_div;
    while (current_iter <= max_iter)
    {
        current_iter++;
        b.setZero();
        p.setZero(); // solution matrix
        A.setZero();
        A.reserve(Eigen::VectorXi::Constant(c.n_particles, 600));

        // A.reserve(VectorXi::Constant(c.n_particles,1000));
        // std::cout<<"Here0"<<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        // std::cout<< "ptype(1): Ptype(1,1) = "<< p_type(1)<<":"<<(1,0)<< std::endl;
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i<c.n_particles; i++)
        {
            if (p_type(i)==0)
                continue;
            // std::cout<< "i: "<<i<<std::endl;
            for (unsigned int j=0; j<nearIndex[i].size(); j++)
            {
                if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
                {
                    MatrixXX r_ij(1,2);
                    MatrixXX v_ji(1,2);
                    MatrixXX temp_mat(1,2);
                    data_type temp;
                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    v_ji = vel.row(nearIndex[i][j]) - vel.row(i);

                    // data_type lap = lap_poly6(nearDist[i][j], c);
                    data_type lap = laplacian.coeff(i, nearIndex[i][j]);

                    A.insert(i, nearIndex[i][j]) = lap*c.mass/density(i);
                    // temp_mat = gradient_poly6(nearDist[i][j], c, r_ij);
                    temp_mat(0,0) = gradient_x.coeff(i, nearIndex[i][j]);
                    temp_mat(0,1) = gradient_y.coeff(i, nearIndex[i][j]);
                    temp = v_ji.row(0).dot(temp_mat.row(0));
                    b(i) = b(i) + temp*c.mass/density(i);
                }
            }
        }
        // std::cout<< "Here 1-1"<< std::endl;
#pragma omp barrier
        // exit(1);
        writeMatrixToFile(b, "b.csv");
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (current_iter%10 == 0) std::cerr << ">>Time taken to prepare part of A: " << duration.count()/1e6 << " seconds\n";
        start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i< c.n_particles; i++)
        {

            data_type sum = 0;
            for (unsigned int j=0; j<nearIndex[i].size(); j++)
            {
                if (j == i)
                    continue;
                sum = sum - A.coeff(i, nearIndex[i][j]);
            }
            A.insert(i, i) = sum;
        }
#pragma omp barrier
        data_type this_sum_temp = A.cwiseProduct(A).sum();
        int numberOfNonZeroElements = A.nonZeros();

        std::cout << "Number of non-zero elements: " << numberOfNonZeroElements << std::endl;
        std::cout<< "Sum of A: "<< this_sum_temp<< std::endl;
        this_sum_temp = b.cwiseProduct(b).sum();
        std::cout<< "Sum of b: "<< this_sum_temp<< std::endl;
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (current_iter%10 == 0) std::cerr << ">>Time taken to prepare the diagonal of A: " << duration.count()/1e6 << " seconds\n";
        

        using namespace Eigen;

        std::cout<< "Dimensions of A: "<< A.rows() << "x" << A.cols() << std::endl;
        std::cout<< "Dimensions of b: "<< b.rows()  << std::endl;
        start = std::chrono::high_resolution_clock::now();
        ConjugateGradient<SparseMatrix<data_type>, Lower|Upper> cg;
        cg.setMaxIterations(1000);
        cg.setTolerance(1e-5);
        cg.compute(A);
        p = cg.solve(b);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout<< "here"<<std::endl;
        if (current_iter%1 == 0)
        {
            std::cerr << ">>Time taken to solve the system: " << duration.count()/1e6 << " seconds\n";
            std::cout << "#iterations:     " << cg.iterations() << std::endl;
            std::cout << "estimated error: " << cg.error()      << std::endl;
        }
        MatrixXX q(c.n_particles, 2);
        std::cout<<"size of vel: "<< vel.size()<<" and of q is: "<< q.size() << std::endl;
        q = cal_div_part_vel(pos, density, p_type, nearIndex, nearDist, p, gradient_x, gradient_y, c);
        vel = vel - q;
        calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, gradient_x, gradient_y, c);
        max_div = divergence.maxCoeff();
        if (current_iter%10 == 0)
        {
            std::cerr << "Current iteration: " << current_iter << " with the max divergence being " << max_div<< std::endl;
            writeMatrixToFile(vel, std::to_string(current_iter)+"_vel.csv");
        }
        if (max_div < 1e-5)
        {
            break;
        }
    }
    if (current_iter == max_iter-1)
    {
        std::cerr << "Pressure Poisson Solver did not converge with the max divergence being " << max_div<< std::endl;

    }
    else
    {
        std::cerr << "Pressure Poisson Solver converged at "<< current_iter<<" with the max divergence being " << max_div<< std::endl;
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
    for (unsigned int i = 0; i<c.n_particles; i++)
    {
        if (p_type(i) == 1)
        {
            for (unsigned int j=0; j<nearIndex[i].size(); j++)
            {
                if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
                {

                    // std::cout<< "i: "<< i<< ";  j:  "<< j<< std::endl;
                    // std::cerr<<"Here0"<< std::endl;
                    MatrixXX r_ij(1,2);
                    MatrixXX weight(1,2);
                    // std::cerr<<"Here0"<< std::endl;
                    weight.fill(0);
                    MatrixXX temp_mat(1,2);
                    data_type temp;

                    // std::cerr<<"Here1"<< std::endl;
                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    // std::cout<< "r_ij: "<< r_ij << std::endl;
                    // std::cerr<<"Here2"<< std::endl;
                    weight(0,0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0,1) = gradient_y.coeff(i, nearIndex[i][j]);
                    // std::cout<< "weight: "<< weight << std::endl;
                    // std::cerr<<"Here3"<< std::endl;
                    // std::cout<< "p.row(nearIndex[i][j]):    "<< p.row(nearIndex[i][j])<<", p.row(i):   "<< p.row(i)<< std::endl;
                    // weight = gradient_poly6(nearDist[i][j], c, r_ij);
                    temp = p(nearIndex[i][j]) - p(i);
                    // temp_mat = p.row(nearIndex[i][j]) - p.row(i);
                    // std::cout<< "temp_mat: "<< temp << std::endl;
                    // weight = weight.cwiseProduct(temp_mat);
                    // std::cout<< "weight: "<< weight << std::endl;
                    // weight = weight.row(0).dot(temp_mat.row(0));
                    weight = weight *temp;
                    temp_mat = weight*c.mass/density(nearIndex[i][j]);
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

