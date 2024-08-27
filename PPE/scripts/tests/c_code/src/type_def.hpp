#ifndef TYPE_DEF_HPP
#define TYPE_DEF_HPP
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#define data_type double // data type for the particles // Need double for Knn, change accordingly. 



struct constants
{
    data_type h;
    data_type dp;
    data_type h_fac;
    data_type mass;
    data_type boundary_size;
    data_type x_y_bn;
    data_type x_y_bp;
    data_type x_y_n;
    data_type x_y_p;
    unsigned int resolution;
    unsigned int n_particles;
    unsigned int mid_idx;
    data_type Eta;
    data_type radius; // kh, radius of influence
    data_type ker_fac;
};


typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> MatrixXX;
typedef Eigen::Vector<data_type, Eigen::Dynamic> VectorX;
typedef Eigen::SparseMatrix<data_type, Eigen::RowMajor> SpMatrixXX;

constants define_constants(data_type size, data_type dp, data_type boundary_fac);
void make_particles(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type, MatrixXX &normals);

#endif