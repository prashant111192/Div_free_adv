#include "compute.hpp"
#include <iostream>
inline void scale_vector(const std::vector<data_type>& input, std::vector<data_type>& output, const data_type scale) {
    std::transform(input.begin(), input.end(), output.begin(),
                   [scale](data_type x) { return x * scale; });
}

inline void multiply_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x * y; });
}
inline void sub_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x - y; });
}
inline void add_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x + y; });
}

MatrixXX gradient_poly6(const data_type &distance, const data_type kh, const MatrixXX &r_ij)
{
    MatrixXX grad(2,1);
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
                const MatrixXX &vel,
                const MatrixXX &density,
                const MatrixXi &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                MatrixXX &divergence,
                const constants &c)
{
    
// #pragma omp parallel for num_threads(10)
    for(unsigned int i = 0; i<c.n_particles; i++)
    {
        if (p_type(i) == 1) // if Fluid particle
        {
            for(unsigned int j=0; j<nearIndex[i].size(); j++)
            {
               if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
            //    if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && p_type[nearIndex[i][j]] == 1)
                {
                    MatrixXX r_ij(2,1);
                    MatrixXX weight(2,1);
                    weight.fill(0);
                    MatrixXX v_ji(2,1);

                    r_ij = pos.row(i) - pos.row(nearIndex[i][j]);
                    v_ji = vel.row(nearIndex[i][j]) - vel.row(i);
                    weight = gradient_poly6(nearDist[i][j], c.radius, r_ij);
                    data_type temp = weight.col(0).dot(v_ji.col(0));
                    std::cout<< pos.row(i) << "\t" << pos.row(nearIndex[i][j]) <<"\t" << r_ij<< "weight: " << weight 
                                << "\ttemp: "<<temp<<  std::endl;

                    // temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]+c.Eta);
                    temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]);
                    divergence(i) += temp;
                }
            }
            divergence(i) = divergence(i)/density(i);
        }
    }
}
