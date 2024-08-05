#include "kernel.hpp"

MatrixXX gradient_poly6(const data_type &distance, const constants &c, const MatrixXX &r_ij)
{
    MatrixXX grad(1,2);
    grad.fill(0.0);
    // auto fac = c.ker_fac;
    if (distance > 0 && distance <= c.radius)
    {
        // CHECK properly pls
        auto temp = c.radius*c.radius - distance*distance;
        auto temp_2 = c.ker_fac*(-6)*temp*temp;
        grad = r_ij * temp_2;
    }
    return grad;
}


data_type lap_poly6(const data_type distance,
                    const constants &c)
{
    // data_type fac = c.ker_fac;
    data_type temp = c.radius*c.radius - distance * distance;
    data_type lap = 0;
    // if (distance>0 && distance<=c.radius)
    // {
        lap = c.ker_fac *(3 * (2 *temp * (4 *distance *distance)+(temp*temp)*(-2)));
    // }
    return lap;
}

void prepare_grad_lap_matrix(const MatrixXX &pos,
                            const std::vector<std::vector<unsigned int>> &nearIndex,
                            const std::vector<std::vector<data_type>> &nearDist,
                            const constants &c,
                            SpMatrixXX &gradient_x,
                            SpMatrixXX &gradient_y,
                            SpMatrixXX &laplacian)
{
// #pragma omp parallel for num_threads(10)
    // testing
    std::cout << "Size of gradient_x: " << gradient_x.nonZeros() << std::endl;
    data_type sum_temp = 0;
    data_type sum_temp_2 = 0;
    data_type max = 10e-20;
    int count = 0;
    int count_nn = 0;   
    // for (unsigned int i = 0; i < 1; i++)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        for (unsigned int j = 0; j < nearIndex[i].size(); j++)
        {
            count_nn += 1;
            if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
            {
                count = count + 1;
                MatrixXX r_ij(1, 2);
                r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                MatrixXX weight(1, 2);
                weight.fill(0);
                weight = gradient_poly6(nearDist[i][j], c, r_ij);
    // testing
                sum_temp += (weight(0,0));
                // sum_temp += abs(weight(0,0));
                data_type temp = weight.row(0).dot(r_ij.row(0));
                gradient_x.insert(i, nearIndex[i][j]) = weight(0, 0);
                gradient_y.insert(i, nearIndex[i][j]) = weight(0, 1);
                laplacian.insert(i, nearIndex[i][j]) = lap_poly6(nearDist[i][j], c);
                sum_temp_2 += gradient_x.coeff(i, nearIndex[i][j]);
                if (max< gradient_x.coeff(i, nearIndex[i][j]))
                {
                    max = gradient_x.coeff(i, nearIndex[i][j]);
                }
                // sum_temp += lap_poly6(nearDist[i][j], c);
            }
        }
    }

    // testing
    std::cout << "Number of nearest neighbours: " << count_nn << std::endl;
    std::cout<< "Number of non zero elements using count variable in gradient_x: " << count << std::endl;
    std::cout << "Non Zero in gradient_x: " << gradient_x.nonZeros() << std::endl;
    std::cout<< "Sizeof c"<< sizeof(c) << std::endl;
    std::cerr<< "Size of gradient_x before compression:"<< sizeof(gradient_x) << std::endl;
    std::cout<< "the sum of the grad_x: " << sum_temp << std::endl;
    std::cout<< "the sum of the grad_x using coeff: " << sum_temp_2 << std::endl;
    std::cout<< "the sum using .sum(): " << gradient_x.sum() << std::endl;


    gradient_x.makeCompressed();
    gradient_y.makeCompressed();
    laplacian.makeCompressed();
    std::cerr<< "Size of gradient_x after compression:"<< sizeof(gradient_x) << std::endl;
    std::cout<< "sum after compressed: " << gradient_x.sum() << std::endl;
    data_type another_sum = 0;
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        for (unsigned int j = 0; j < nearIndex[i].size(); j++)
        {
            if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
            {
                another_sum += abs(gradient_x.coeff(i, nearIndex[i][j]));
            }
        }
    }
    std::cerr<< "Size of gradient_x before compression:"<< sizeof(gradient_x) << std::endl;
    std::cout<< "sum after compressed and using coeff: " << another_sum << std::endl;

    std::cout<< "max using for loop: " << max << std::endl;
    std::cout<< "max coeff: " <<gradient_x.coeffs().maxCoeff() << std::endl;
    std::cout<< "max coeff: " <<gradient_x.coeffs().minCoeff() << std::endl;
    std::cout << "Size of gradient_x: " << gradient_x.nonZeros() << std::endl;
    std::cout << "memory used by gradient_x: " << gradient_x.nonZeros() * sizeof(data_type) / (1024 * 1024) << " Mbytes" << std::endl;
    std::cout << "Size of gradient_x: " << laplacian.nonZeros() << std::endl;
    std::cout << "memory used by gradient_x: " << laplacian.nonZeros() * sizeof(data_type) / (1024 * 1024) << " Mbytes" << std::endl;
    std::cerr<< "Size of gradient_x before compression:"<< sizeof(gradient_x) << std::endl;
}
    