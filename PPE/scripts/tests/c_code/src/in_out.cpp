#include "in_out.hpp"

void make_from_dsph(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type, MatrixXX &pressure)
{

    LOG(INFO) << "Reading DSPH file";
    std::string path("./DSPHdata/");
    std::string ext(".csv");
    for (auto &file : std::filesystem::directory_iterator(path))
    {
        if  (file.path().extension() == ext)
        {
            std::string filename = file.path().string();
            // std::string filename = file.path().filename().string();
            std::ifstream inFile(filename);
            LOG(INFO) << filename;
            std::string datLineStr;
            long int current_line = 0;
            while(std::getline(inFile, datLineStr))
            {
                if (current_line > 2)
                {
                    std::stringstream datLine(datLineStr);
                    std::string data;
                    std::vector<data_type> dataVec;
                    while(std::getline(datLine, data, ';'))
                    {
                        dataVec.push_back(std::stod(data));
                    }
                    pos(current_line-3, 0) = dataVec[0];
                    pos(current_line-3, 1) = dataVec[2];

                    // if (dataVec[0] > -.5 && dataVec[0] < 0.5 && dataVec[2] > -10 && dataVec[2] < 0 && dataVec[8] ==3)
                    // {
                    //     vel(current_line-3, 0) = 0;
                    //     vel(current_line-3, 1) = 1.5;
                    // }
                    // else{
                    //     vel(current_line-3, 0) = 0;
                    //     vel(current_line-3, 1) = 0;
                    // }
                    vel(current_line-3, 0) = dataVec[4];
                    vel(current_line-3, 1) = dataVec[6];
                    if (dataVec[8] == 3) // Fluid
                        p_type(current_line-3, 0) = 1;
                    else                 // Boundary
                        p_type(current_line-3, 0) = 0;

                    if (dataVec[7] > 1020)
                        density(current_line-3, 0) = 1020;
                        else
                        density(current_line-3, 0) = dataVec[7];
                    // if (dataVec[0] > -0.1 && dataVec[0] < 0.1 && dataVec[2] > -0.1 && dataVec[2] < 0.1)
                    // {
                    //     std::cout<< "position:" << dataVec[0] << " " << dataVec[2] << std::endl;
                    //     std::cout<< "velocity:" << dataVec[4] << " " << dataVec[6] << std::endl;
                    //     std::cout<< "density:" << dataVec[7] << std::endl;
                    //     std::cout<< "type:" << dataVec[8] << std::endl;
                    // }
                    // if (pos(current_line-3, 0) < 0.05 && pos(current_line-3, 0) > -0.05 && pos(current_line-3, 1) < 0.05 && pos(current_line-3, 1) > -0.05 && 1==0)
                    //     {
                    //         vel(current_line-3, 0) = 0;
                    //         vel(current_line-3, 1) = 0;
                    //         p_type(current_line-3, 0) = 0;

                    //     }
                    //     else{

                    //         vel(current_line-3, 0) = dataVec[4];
                    //         vel(current_line-3, 1) = dataVec[6];
                    //         if (dataVec[8] == 3) // Fluid
                    //             p_type(current_line-3, 0) = 1;
                    //         else                 // Boundary
                    //             p_type(current_line-3, 0) = 0;
                    //     }
                    // if (dataVec[7] > 1020)
                    //     density(current_line-3, 0) = 1020;
                    //     else
                    //     density(current_line-3, 0) = dataVec[7];
                    // pressure(current_line-3, 0) = dataVec[8];
                }
                else{
                    std::cout<< "current line: " << current_line <<":" << datLineStr << std::endl;
                }
                current_line ++;
            }
        }
    }
}