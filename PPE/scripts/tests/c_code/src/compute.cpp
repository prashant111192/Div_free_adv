#include <iostream>
#include "compute.hpp"

MatrixXX gradient_poly6(const data_type &distance, const data_type kh, const MatrixXX &r_ij)
{
    MatrixXX grad(1,2);
    grad.fill(0);
    data_type h1 = 1/kh;
    auto fac = (4*h1*h1*h1*h1*h1*h1*h1*h1)*(M_1_PI);
    auto temp = kh*kh - distance*distance;
    if (0 < distance && distance <= kh)
    {
        auto temp_2 = fac*(-6)*temp*temp;
        grad = r_ij * temp_2;
    }
    return grad;
}

void calc_divergence(const MatrixXX &pos,
                MatrixXX &vel,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                MatrixXX &divergence,
                const constants &c)
{
    
#pragma omp parallel for num_threads(10)
    for(unsigned int i = 0; i<c.n_particles; i++)
    {
        if (p_type(i) == 1) // if Fluid particle
        {
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
                    weight = gradient_poly6(nearDist[i][j], c.radius, r_ij);
                    data_type temp = weight.row(0).dot(v_ji.row(0));
                    // std::cout<< pos.row(i) << "\t" << pos.row(nearIndex[i][j]) <<"\t" << r_ij<< "weight: " << weight 
                                // << "\ttemp: "<<temp<<  std::endl;

                    // temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]+c.Eta);
                    temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]);
                    divergence(i) = divergence(i)+ temp;
                }
            }
            divergence(i) = divergence(i)/density(i);
        }
    }
}

void pressure_poisson(const MatrixXX &pos,
                MatrixXX &vel,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                MatrixXX &divergence,
                const constants &c)
{
    int max_iter = 10;
    int current_iter = 0;
    SpMatrixXX A(c.n_particles, c.n_particles);
    A.reserve(VectorXi::Constant(600,c.n_particles));
    MatrixXX b(c.n_particles, 1);
    MatrixXX p(c.n_particles, 1);


    std::cout<<"Here-1"<<std::endl;
    data_type max_div;
    while (current_iter < max_iter)
    {
        b.fill(0);
        A.resize(0,0);
        A.reserve(VectorXi::Constant(600,c.n_particles));
        // A.reserve(VectorXi::Constant(c.n_particles,1000));

        std::cout<<"Here0"<<std::endl;
#pragma omp parallel for num_threads(10)
        for (unsigned int i = 0; i<c.n_particles; i++)
        {
            if (p_type(i)==0)
                continue;
            // std::cout<< "i: "<<i<<std::endl;
            for (unsigned int j=0; j<nearIndex[i].size(); j++)
            {
                if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && i!=j)
                {
                    MatrixXX r_ij(1,2);
                    MatrixXX v_ji(1,2);
                    MatrixXX temp_mat(1,2);
                    data_type temp;
                    // std::cout<< "pos.row(i): "<<pos.row(i) << "\t pos.row(nearIndex[i][j]): "<<pos.row(nearIndex[i][j]) << "\t r_ij: "<<r_ij<<std::endl;
                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    // std::cout<< "vel.row(nearIndex[i][j]): "<<vel.row(nearIndex[i][j]) << "\t vel.row(i): "<<vel.row(i) << "\t v_ji: "<<v_ji<<std::endl;
                    v_ji = vel.row(nearIndex[i][j]) - vel.row(i);

                    data_type lap = lap_poly6(nearDist[i][j], c.radius);

                    std::cerr<<"Here0.1"<<std::endl;
                    A.insert(i, nearIndex[i][j]) = lap*c.mass/density(i);
                    std::cerr<<"Here0.2"<<std::endl;
                    temp_mat = gradient_poly6(nearDist[i][j], c.radius, r_ij);
                    std::cerr<<"Here0.3"<<std::endl;
                    temp = v_ji.row(0).dot(temp_mat.row(0));
                    std::cerr<<"Here0.4"<<std::endl;
                    b(i) = b(i) + temp*c.mass/density(i);
                    std::cerr<<"Here0.5"<<std::endl;
                }
                A.insert(i,i) = -1*A.row(i).sum();
            }
        }
        std::cout<<"Here1"<<std::endl;
        

        ConjugateGradient<SparseMatrix<data_type>, Lower|Upper> cg;
        cg.setMaxIterations(10);
        cg.setTolerance(1e-5);
        cg.compute(A);
        p = cg.solve(b);
        std::cout << "#iterations:     " << cg.iterations() << std::endl;
        std::cout << "estimated error: " << cg.error()      << std::endl;
        MatrixXX q = cal_div_part_vel(pos, density, p_type, nearIndex, nearDist, p, c);
        vel = vel - q;
        current_iter++;
        calc_divergence(pos, vel, density, p_type, nearIndex, nearDist, divergence, c);
        max_div = divergence.maxCoeff();
        if (current_iter%10 == 0)
        {
            std::cerr << "Current iteration: " << current_iter << " with the max divergence being " << max_div<< std::endl;
            writeMatrixToFile(pos, std::to_string(current_iter)+"_pos.csv");
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
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                const MatrixXX &p,
                const constants &c)
{
    MatrixXX q(c.n_particles, 2);
    q.fill(0);
    for (unsigned int i = 0; i<c.n_particles; i++)
    {
        if (p_type(i) == 1)
        {
            for (unsigned int j=0; j<nearIndex[i].size(); j++)
            {
                if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
                {
                    MatrixXX r_ij(1,2);
                    MatrixXX weight(1,2);
                    weight.fill(0);
                    MatrixXX temp_mat(1,2);

                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    weight = gradient_poly6(nearDist[i][j], c.radius, r_ij);
                    weight = weight * (q.row(nearIndex[i][j]) - q.row(i));
                    q.row(i) = q.row(i) + weight.row(0)*c.mass/density(nearIndex[i][j]);
                }
            }
        }
    }
    return q;
}

data_type lap_poly6(const data_type distance,
                    const data_type kh)
{
    data_type h1 = 1/kh;
    data_type fac = (4*h1*h1*h1*h1*h1*h1*h1*h1* M_1_PI);
    data_type temp = kh*kh - distance * distance;
    data_type lap = 0;
    if (distance>0 && distance<=kh)
    {
        lap = fac *(2 *temp * (4 *distance *distance)+(temp*temp*-2));
    }
    return lap;
}
