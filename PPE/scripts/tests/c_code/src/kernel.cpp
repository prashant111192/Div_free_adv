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
        // testing, remove asap
        // temp_2 = 2;
        // MatrixXX grad2(1,2);
        // grad2.fill(5.0);
        grad = r_ij * temp_2;

        // std::cout<< "grad: " << grad << std::endl;

    }
    return grad;
}


data_type lap_poly6(const data_type distance,
                    const constants &c)
{
    // data_type fac = c.ker_fac;
    data_type temp = c.radius*c.radius - distance * distance;
    data_type lap;
    // if (distance>0 && distance<=c.radius)
    // {
    lap = c.ker_fac * (3 * (2 * temp * (4 * distance * distance) + (temp * temp * -2)));
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
    LOG(INFO) << "Preparing the gradient and laplacian matrix";
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(10)
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        for (unsigned int j = 0; j < nearIndex[i].size(); j++)
        {
            // if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
            // {
                MatrixXX r_ij(1, 2);
                r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                MatrixXX weight(1, 2);
                weight.fill(0);
                weight = gradient_poly6(nearDist[i][j], c, r_ij);
                gradient_x.insert(i, nearIndex[i][j]) = weight(0);
                gradient_y.insert(i, nearIndex[i][j]) = weight(1);
                laplacian.insert(i, nearIndex[i][j]) = lap_poly6(nearDist[i][j], c);
            // }
        }
    }

    gradient_x.makeCompressed();
    gradient_y.makeCompressed();
    laplacian.makeCompressed();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "Time taken to prepare the gradient and laplacian matrix: " << duration.count() / 1e6 << " seconds";
}
    