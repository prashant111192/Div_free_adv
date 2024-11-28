#include "kernel.hpp"

MatrixXX gradient_poly6(const data_type &distance, const constants &c, const MatrixXX &r_ij)
{
    MatrixXX grad(1, 2);
    grad.fill(0.0);
    // auto fac = c.ker_fac;
    if (distance > 0 && distance <= c.radius)
    {
        // CHECK properly pls
        auto temp = c.radius * c.radius - distance * distance;
        auto temp_2 = c.ker_fac * (-6) * temp * temp;
        // testing, remove asap
        // temp_2 = 2;
        // MatrixXX grad2(1,2);
        // grad2.fill(5.0);
        grad = r_ij * temp_2;

        // std::cout<< "grad: " << grad << std::endl;
    }
    return grad;
}

data_type ker_poly6(const data_type distance, 
                    const constants &c)
{
    data_type val = c.radius * c.radius - distance * distance;
    val = c.ker_fac *val*val*val;
    return val;
}
data_type lap_poly6(const data_type distance,
                    const constants &c)
{
    // data_type fac = c.ker_fac;
    data_type temp = c.radius * c.radius - distance * distance;
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
            MatrixXX r_ij(1, 2);
            r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
            MatrixXX weight(1, 2);
            weight.fill(0);
            weight = gradient_poly6(nearDist[i][j], c, r_ij);
// Potentially, the insert from different threads is writing to the same memory address as its a sparse matrix.
// Maybe, can be fixed by specifying the total size of the matrix. NOPE, that didn't work. FML
// Maybe can  be fixed by using a mutex lock. a CHATGPT suggestion
#pragma omp critical(foo1)
            gradient_x.insert(i, nearIndex[i][j]) = weight(0);
#pragma omp critical(foo2)
            gradient_y.insert(i, nearIndex[i][j]) = weight(1);
#pragma omp critical(foo3)
            laplacian.insert(i, nearIndex[i][j]) = lap_poly6(nearDist[i][j], c);
        }
    }

    gradient_x.makeCompressed();
    gradient_y.makeCompressed();
    laplacian.makeCompressed();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    CLOG(INFO, "TIME") << "Preparing the gradient and laplacian matrix(s):" << duration.count() / 1e6;
}



void prepare_grad_lap_matrix_fast(const MatrixXX &pos,
                             const std::vector<std::vector<unsigned int>> &nearIndex,
                             const std::vector<std::vector<data_type>> &nearDist,
                             const constants &c,
                             SpMatrixXX &ker_vals,
                             SpMatrixXX &gradient_x,
                             SpMatrixXX &gradient_y,
                             SpMatrixXX &laplacian,
                             int max_NN)
{
    LOG(INFO) << "Preparing the gradient and laplacian matrix";
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Eigen::Triplet<data_type>> tripletList_x, tripletList_y, tripletList_lap, tripletList_ker_vals;
    tripletList_ker_vals.reserve(c.n_particles * max_NN);
    tripletList_x.reserve(c.n_particles * max_NN);  // Estimate reserve size based on particle connections
    tripletList_y.reserve(c.n_particles * max_NN);
    tripletList_lap.reserve(c.n_particles * max_NN);

#pragma omp parallel for
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        std::vector<Eigen::Triplet<data_type>> local_tripletList_x, local_tripletList_y, local_tripletList_lap, local_tripletList_ker_vals;
        local_tripletList_ker_vals.reserve(nearIndex[i].size());
        local_tripletList_x.reserve(nearIndex[i].size());
        local_tripletList_y.reserve(nearIndex[i].size());
        local_tripletList_lap.reserve(nearIndex[i].size());

        for (unsigned int j = 0; j < nearIndex[i].size(); j++)
        {
            MatrixXX r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
            MatrixXX weight = gradient_poly6(nearDist[i][j], c, r_ij);
            data_type ker_val = ker_poly6(nearDist[i][j], c);
            

            local_tripletList_ker_vals.emplace_back(i, nearIndex[i][j], ker_val);
            local_tripletList_x.emplace_back(i, nearIndex[i][j], weight(0));
            local_tripletList_y.emplace_back(i, nearIndex[i][j], weight(1));
            local_tripletList_lap.emplace_back(i, nearIndex[i][j], lap_poly6(nearDist[i][j], c));
        }

#pragma omp critical
        {
            tripletList_ker_vals.insert(tripletList_ker_vals.end(), local_tripletList_ker_vals.begin(), local_tripletList_ker_vals.end());
            tripletList_x.insert(tripletList_x.end(), local_tripletList_x.begin(), local_tripletList_x.end());
            tripletList_y.insert(tripletList_y.end(), local_tripletList_y.begin(), local_tripletList_y.end());
            tripletList_lap.insert(tripletList_lap.end(), local_tripletList_lap.begin(), local_tripletList_lap.end());
        }
    }

    ker_vals.setFromTriplets(tripletList_ker_vals.begin(), tripletList_ker_vals.end());
    gradient_x.setFromTriplets(tripletList_x.begin(), tripletList_x.end());
    gradient_y.setFromTriplets(tripletList_y.begin(), tripletList_y.end());
    laplacian.setFromTriplets(tripletList_lap.begin(), tripletList_lap.end());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    CLOG(INFO, "TIME") << "Preparing the gradient and laplacian matrix took (s):" << duration.count() / 1e6 ;
}
