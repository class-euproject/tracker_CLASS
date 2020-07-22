#include "Tracker.h"
#include "Tracking.h"
#include <time.h>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace tracking;

namespace py = pybind11;

using namespace tracking;

// TODO: how to update trackers in tracking to take into account the info already tracked
// TODO: x, y -> is it latitude, longitude?
std::tuple<std::vector<Tracker>, std::vector<bool>, int> track2(std::vector<obj_m>& frame, float dt, int n_states, int initial_age, int age_threshold, std::vector<Tracker> &trackers, std::vector<bool> &trackerIndexes, int curIndex)
{
    /*geodetic_converter::GeodeticConverter gc;
    gc.initialiseReference(44.655540, 10.934315, 0);
    double east, north, up;

    int it = 0;
    for (auto t : frame){
    gc.geodetic2Enu(t.x_, t.y_, 0, &east, &north, &up);
    t.x_ = east;
    t.y_ = north;
    frame[it++] = t;
    }*/
    // NOT NEEDED AS DATA FROM YOLO ALREADY IN METERS FORMAT


    Tracking tracking(n_states, dt, initial_age);
    tracking.setAgeThreshold(age_threshold);
    tracking.trackers = trackers;
    /** TODO by BSC: Changed from private to public */
    for (int i = 0; i < trackerIndexes.size(); i++) {
        tracking.trackerIndexes[i] = trackerIndexes[i];
    }
    // tracking.trackerIndexes = trackerIndexes;
    tracking.curIndex = curIndex;
    /** **/
    tracking.track(frame);
    std::vector<bool> newTrackerIndexes(MAX_INDEX);
    for (int i = 0; i < MAX_INDEX; i++) {
        newTrackerIndexes[i] = tracking.trackerIndexes[i];
    }
    auto tuple = std::make_tuple(tracking.getTrackers(), newTrackerIndexes, tracking.curIndex);
    return tuple; // TODO: getters needed
    //return tracking.getTrackers();
}


PYBIND11_MODULE(track, m) {

    py::class_<obj_m>(m, "obj_m")
            .def(py::init<const float, const float, const int, const int, const int, const int>())
            .def_readwrite("x", &obj_m::x)
            .def_readwrite("y", &obj_m::y)
            .def_readwrite("frame", &obj_m::frame)
            .def_readwrite("cl", &obj_m::cl)
            .def_readwrite("w", &obj_m::w)
            .def_readwrite("h", &obj_m::h)
            .def(py::pickle(
                    [](py::object self) {
                        return py::make_tuple(self.attr("x"), self.attr("y"), self.attr("frame"), self.attr("cl"), self.attr("w"), self.attr("h"));
                    },
                    [](const py::tuple &t) {
                        if (t.size() != 6)
                            throw std::runtime_error("Invalid state!");
                        float x       = t[0].cast<float>();
                        float y       = t[1].cast<float>();
                        int frame     = t[2].cast<int>();
                        int classO    = t[3].cast<int>();
                        int w         = t[4].cast<int>();
                        int h         = t[5].cast<int>();

                        obj_m data = obj_m(x, y, frame, classO, w, h);
                        return data;
                    }
            ));
    py::class_<state>(m, "state")
            .def(py::init<const float, const float, const float, const float, const float>())
            .def_readwrite("x", &state::x)
            .def_readwrite("y", &state::y)
            .def_readwrite("yaw", &state::yaw)
            .def_readwrite("vel", &state::vel)
            .def_readwrite("yawRate", &state::yawRate)
            .def(py::pickle(
                    [](py::object self) {
                        return py::make_tuple(self.attr("x"), self.attr("y"), self.attr("yaw"), self.attr("vel"), self.attr("yawRate"));
                    },
                    [](const py::tuple &t) {
                        if (t.size() != 5)
                            throw std::runtime_error("Invalid state!");
                        float x       = t[0].cast<float>();
                        float y       = t[1].cast<float>();
                        float yaw     = t[2].cast<float>();
                        float vel     = t[3].cast<float>();
                        float yaw_rate= t[4].cast<float>();

                        state st = state(x, y, yaw, vel, yaw_rate);
                        return st;
                    }
            ));

    py::class_<EKF>(m, "EKF")
            .def(py::init<>())
            .def_readwrite("nStates", &EKF::nStates)
            .def_readwrite("dt", &EKF::dt)
            .def_readwrite("xEst", &EKF::xEst)
            .def_readwrite("Q", &EKF::Q)
            .def_readwrite("R", &EKF::R)
            .def_readwrite("P", &EKF::P)
            .def_readwrite("H", &EKF::H)
            .def(py::pickle(
                    [](py::object self) {
                        return py::make_tuple(self.attr("nStates"), self.attr("dt"), self.attr("xEst"), self.attr("Q"), self.attr("R"), self.attr("P"), self.attr("H"));
                    },
                    [](const py::tuple &t) {
                        if (t.size() != 7)
                            throw std::runtime_error("Invalid state!");
                        int nStates  		= t[0].cast<int>();
                        float dt     		= t[1].cast<float>();
                        state xEst   		= t[2].cast<state>();
                        Eigen::Matrix<float, -1, -1> Q	        = t[3].cast<Eigen::Matrix<float, -1, -1>>();
                        Eigen::Matrix<float, -1, -1> R	        = t[4].cast<Eigen::Matrix<float, -1, -1>>();
                        Eigen::Matrix<float, -1, -1> P         = t[5].cast<Eigen::Matrix<float, -1, -1>>();
                        Eigen::Matrix<float, -1, -1> H         = t[6].cast<Eigen::Matrix<float, -1, -1>>();

                        EKF ekf 	= EKF();
                        ekf.nStates 	= nStates;
                        ekf.dt   	= dt;
                        ekf.xEst	= xEst;
                        ekf.Q		= Q;
                        ekf.R		= R;
                        ekf.P		= P;
                        ekf.H		= H;
                        return ekf;
                    }
            ));

    py::class_<Tracker>(m, "Tracker")
            .def(py::init<const obj_m&, const int, const float, const int, const int>())
            .def_readwrite("traj", &Tracker::traj)
            .def_readwrite("zList", &Tracker::zList)
            .def_readwrite("predList", &Tracker::predList)
            .def_readwrite("ekf", &Tracker::ekf)
            .def_readwrite("age", &Tracker::age)
            .def_readwrite("r", &Tracker::r)
            .def_readwrite("g", &Tracker::g)
            .def_readwrite("b", &Tracker::b)
            .def_readwrite("cl", &Tracker::cl)
            .def_readwrite("id", &Tracker::id)
            .def(py::pickle(
                    [](py::object self){
                        return py::make_tuple(self.attr("traj"), self.attr("zList"), self.attr("predList"), self.attr("ekf"),self.attr("age"), self.attr("r"), self.attr("g"), self.attr("b"), self.attr("cl"), self.attr("id"));
                    },
                    [](const py::tuple &t){
                        std::vector<obj_m> traj = t[0].cast<std::vector<obj_m>>();
                        std::vector<state> zList = t[1].cast<std::vector<state>>();
                        std::vector<state> predList = t[2].cast<std::vector<state>>();
                        EKF ekf = t[3].cast<EKF>();
                        int age = t[4].cast<int>();
                        int r = t[5].cast<int>();
                        int g = t[6].cast<int>();
                        int b = t[7].cast<int>();
                        int classO = t[8].cast<int>();
                        int id = t[9].cast<int>();
                        Tracker tracker(traj, zList, predList, ekf, age, r, g, b, classO, id);
                        return tracker;
                    }
            ));

    m.def("track2", &track2, py::return_value_policy::automatic);
}