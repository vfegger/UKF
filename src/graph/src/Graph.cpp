#include <iostream>
#include "../include/Gnuplot.hpp"

int main(){
    std::cout << "\nStart Execution\n\n";
    std::string path_text_in = "/mnt/d/Research/UKF/data/text/out/";
    std::string path_text_out = "/mnt/d/Research/UKF/graph/data/";

    std::string extension_text_in = ".dat";
    std::string extension_text_out = ".dat";

    GnuplotParser::ConvertToGnuplot(path_text_in,path_text_out,extension_text_in,extension_text_out);

    std::cout << "\nEnd Execution\n";
    return 0;
}