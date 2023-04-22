#include <iostream>
#include "../include/Gnuplot.hpp"

int main()
{
    std::cout << "\nStart Execution\n\n";
    std::string path_dir = std::filesystem::current_path().string();
    std::string path_text_in = path_dir + "/graph/data/in/";
    std::string path_text_out = path_dir + "/graph/data/out/";

    std::string extension_text_in = ".dat";
    std::string extension_text_out = ".dat";

    GnuplotParser::ConvertToGnuplot(path_text_in, path_text_out, extension_text_in, extension_text_out);

    std::cout << "\nEnd Execution\n";
    return 0;
}