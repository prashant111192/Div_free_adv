#ifndef IN_OUT_HPP
#define IN_OUT_HPP
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <cstring>

#include "type_def.hpp"

void make_from_dsph(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type, MatrixXX &pressure);

template <typename T>
void writeMatrixToFile(const MatrixXX &pos, T &x, std::string filename)
    {
    // int precision = 6;
    filename = filename + ".csv";
    LOG(INFO)<< "Writing to CSV: "<< filename;
    auto chrono_start = std::chrono::high_resolution_clock::now();
    std::ofstream file(filename, std::ios::out);
    if (file.is_open()) {
        // file << std::fixed << std::setprecision(precision);
        for (int i = 0; i < x.rows(); ++i) {
            file<< pos(i,0) << "," << pos(i,1) << ",";
            for (int j = 0; j < x.cols(); ++j) {
                file << x(i, j);
                if (j < x.cols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        SSD
        LOG(INFO) << "Matrix written to " << filename;
    } else {
        LOG(ERROR) << "Unable to open file " << filename;
    }
    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count();
    LOG(INFO) << "Time taken : " << duration << " ms\n";
}

template <typename T>
void writeMatrixToBinaryFile(const MatrixXX &pos, T &x, const std::string &filename) {
    filename = filename + ".bin";
    LOG(INFO) << "Writing to binary: " << filename;
    auto chrono_start = std::chrono::high_resolution_clock::now();
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        int rows = x.rows();
        int cols = x.cols()+pos.cols();
        // std::cout<< "pos cols: " << pos.cols() << std::endl;
        // std::cout<< "Name of file: " << filename << " rows: " << rows << " cols: " << cols << std::endl;
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        
        // Write position data
        for (int i = 0; i < pos.rows(); ++i) {
            data_type posData[2] = { pos(i, 0), pos(i, 1) };
            file.write(reinterpret_cast<const char*>(posData), 2 * sizeof(data_type));
            
            // Write x matrix data for each row
            file.write(reinterpret_cast<const char*>(&x(i, 0)), cols * sizeof(x(i, 0)));
        }
        
        file.close();
        LOG(INFO) << "Matrix written to " << filename;
    } 
    else {
        LOG(ERROR) << "Unable to open file " << filename;
    }
    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count();
    LOG(INFO) << "Time taken: " << duration << " ms and the max size should be " << x.rows() * x.cols() * sizeof(data_type) +(x.rows() * 2 * sizeof(data_type)) + 2 * sizeof(int) << " bytes\n";
}

#endif