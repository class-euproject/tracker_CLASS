#ifndef TRACKUTILS_H
#define TRACKUTILS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "geodetic_conv.hpp"

namespace tracking{
/*Object in meters*/
struct obj_m{
    double x      = 0;
    double y      = 0;
    int frame     = -1;
    int cl        = -1;
    int w         = 0;
    int h         = 0;
    double lat    = 0;
    double lon    = 0;
    int pixel_x   = 0;
    int pixel_y   = 0;

    obj_m(){}
    obj_m(const double x_, const double y_, const int frame_, const int cl_, const int width, const int height,
          const double lat_, const double lon_, const int pixel_x_, const int pixel_y_) : x(x_), y(y_), frame(frame_),
          cl(cl_), w(width), h(height), lat(lat_), lon(lon_), pixel_x(pixel_x_), pixel_y(pixel_y_) {}
    void print();
};
}

std::vector<tracking::obj_m> readDataFromFile(const std::string filename);

#endif /*TRACKUTILS_H*/