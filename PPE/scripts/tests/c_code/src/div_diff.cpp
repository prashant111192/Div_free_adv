#include "div_diff.hpp"
void div_diff_compute(const MatrixXX &pos,
                      MatrixXX &vel,
                      const MatrixXX &density,
                      const Eigen::MatrixXi &p_type,
                      const std::vector<std::vector<unsigned int>> &nearIndex,
                      const std::vector<std::vector<data_type>> &nearDist,
                      MatrixXX &divergence,
                      const Eigen::SparseMatrix<data_type> &gradient_x,
                      const Eigen::SparseMatrix<data_type> &gradient_y,
                      const Eigen::SparseMatrix<data_type> &laplacian,
                      const constants &c,
                      const unsigned int count)
{
    MatrixXX vel_write(c.n_particles, 2);
    vel_write.fill(0);
    #pragma omp parallel for
    for (unsigned int i = 0; i < c.n_particles; i++)
    {
        if (p_type(i) == 1)
        {
            data_type temp = 0;
            data_type diff_coeff_vel = 10e-8;

            for (unsigned int j = 0; j < nearIndex[i].size(); j++)
            {
                if (nearDist[i][j] > 0 && nearDist[i][j] <= c.radius)
                {
                    // data_type div_diff = divergence(i) - divergence(nearIndex[i][j]);
                    // div_diff = div_diff *c.mass /density(i);

                    




                    data_type div_diff = divergence(i) - divergence(nearIndex[i][j]);
                    MatrixXX weight(1, 2);
                    MatrixXX pos_diff(1, 2);
                    pos_diff = pos.row(i) - pos.row(nearIndex[i][j]);
                    weight.fill(0);
                    weight(0, 0) = gradient_x.coeff(i, nearIndex[i][j]);
                    weight(0, 1) = gradient_y.coeff(i, nearIndex[i][j]);
                    
                    data_type dot_prod = weight.row(0).dot(pos_diff.row(0));
                    dot_prod = dot_prod / (pos_diff.squaredNorm() + (c.Eta * c.Eta));
                    dot_prod = dot_prod * c.mass* (density(nearIndex[i][j]) + density(i))/ (density(nearIndex[i][j]) * density(i));
                    dot_prod = dot_prod * diff_coeff_vel * div_diff;
                    temp = temp + dot_prod;
                
                }
            }
            vel_write.row(i) = vel_write.row(i) - (temp *pos.row(i)/(pos.row(i).norm()+(c.Eta * c.Eta)));
            // if (vel.row(i).norm()<=0 && vel(i,0)!=0 && vel(i,1)!=0)
            // { std::cerr<< vel.row(i) << std::endl; }
        }
    }
    vel = (vel_write ) - vel;

}