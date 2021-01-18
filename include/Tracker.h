#ifndef TRACKER_H
#define TRACKER_H

#include "ekf.h"
#include "obj.h"
#include <cstdlib>


namespace tracking {
class Tracker
{
public:
    std::vector<obj_m> traj;
    std::vector<state> zList;
    std::vector<state> predList;
    EKF ekf;
    int age;
    int r;
    int g;
    int b;
    int cl;
    int id;
    int idx; // added by bsc, provides relation with list_boxes

    Tracker(const obj_m &first_point, const int initial_age, const float dt, const int n_states, const int id_);
    Tracker(const std::vector<obj_m>& traj,const EKF& ekf,const int age, const int cl, const int id, const int idx);

    const std::vector<obj_m>& getTraj() const { return traj; };

    const EKF& getEKF() const { return ekf; };

private:
    Tracker();
    EKF ekfInitialize(const float dt, const int n_states, const obj_m &first_point);
};

}

#endif /*TRACKER_H*/