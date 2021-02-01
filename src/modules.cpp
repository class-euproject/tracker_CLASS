#include "Tracker.h"
#include "Tracking.h"
#include <ctime>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "geodetic_conv.hpp"

using namespace tracking;

namespace py = pybind11;

using namespace tracking;


/*std::tuple<std::vector<Tracker>, std::vector<bool>, int, std::vector<std::tuple<float, float, int, uint8_t, uint8_t>>>
        track2(std::vector<obj_m>& frame, std::vector<Tracker> &trackers, std::vector<bool> &trackerIndexes, int curIndex)*/
std::tuple<std::vector<Tracker>, int, std::vector<std::tuple<double, double, int, uint8_t, uint8_t>>>
        track2(std::vector<obj_m>& frame, std::vector<Tracker> &trackers, int curIndex)
{
    int initial_age = -5, age_threshold = -8, n_states = 5;
    float dt = 0.03;

    Tracking tracking(n_states, dt, initial_age);
    tracking.setAgeThreshold(age_threshold);
    tracking.trackers = trackers;
    /** Added by BSC: Changed from private to public */
    tracking.curIndex = curIndex;
    /** **/
    tracking.track(frame);
    /* Return structure to be used in the Deduplicator [lat (float), lon (float), class (int), vel (uint8_t),
     * yaw (uint8_t] */ // TODO: it should be a MasaMessage instead of tuple
    geodetic_converter::GeodeticConverter gc;
    gc.initialiseReference(44.655540, 10.934315, 0);
    std::vector<std::tuple<double, double, int, uint8_t, uint8_t>> infoForDeduplicator;
    double lat, lon, alt;
    uint8_t velocity, yaw;
    for (Tracker t : tracking.getTrackers()) {
	    velocity = yaw = 0;
        //gc.enu2Geodetic(t.traj.back().y, t.traj.back().x, 0, &lat, &lon, &alt); // y -> east, x -> north
        lat = frame[t.idx].lat;
        lon = frame[t.idx].lon;
        std::cout << std::setprecision(10) << "Tracker 20939_" << t.id << ": EAST FIRST north (traj x): " << t.traj.back().x
                  << " east (traj y): " << t.traj.back().y << " lat: " << lat << " lon: " << lon << std::endl;
        if (!t.predList.empty()) {
            velocity = uint8_t(std::abs(t.ekf.xEst.vel * 3.6 * 2));
            yaw = uint8_t((int((t.ekf.xEst.yaw * 57.29 + 360)) % 360) * 17 / 24);
        }
        infoForDeduplicator.emplace_back(lat, lon, t.cl, velocity, yaw);
    }
    return std::make_tuple(tracking.getTrackers(), tracking.curIndex, infoForDeduplicator);
}


PYBIND11_MODULE(track, m) {

    py::class_<obj_m>(m, "obj_m")
            .def(py::init<const double, const double, const int, const int, const int, const int, const double,
                    const double>())
            .def_readwrite("x", &obj_m::x)
            .def_readwrite("y", &obj_m::y)
            .def_readwrite("frame", &obj_m::frame)
            .def_readwrite("cl", &obj_m::cl)
            .def_readwrite("w", &obj_m::w)
            .def_readwrite("h", &obj_m::h)
            .def_readwrite("lat", &obj_m::lat)
            .def_readwrite("lon", &obj_m::lon)
            .def(py::pickle(
                    [](const obj_m& self) {
                        return py::make_tuple(self.x, self.y, self.frame,self.cl, self.w, self.h, self.lat, self.lon);
                    },
                    [](const py::tuple &t) {
                        if (t.size() != 8)
                            throw std::runtime_error("Invalid state!");
                        double x       = t[0].cast<double>();
                        double y       = t[1].cast<double>();
                        int frame     = t[2].cast<int>();
                        int classO    = t[3].cast<int>();
                        int w         = t[4].cast<int>();
                        int h         = t[5].cast<int>();
                        double lat    = t[6].cast<double>();
                        double lon    = t[7].cast<double>();

                        obj_m data = obj_m(x, y, frame, classO, w, h, lat, lon);
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
                    [](const state& self) {
                        return py::make_tuple(self.x, self.y, self.yaw, self.vel, self.yawRate);
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
            //.def_readwrite("Q", &EKF::Q)
            //.def_readwrite("R", &EKF::R)
            .def_readwrite("P", &EKF::P)
            //.def_readwrite("H", &EKF::H)
            .def(py::pickle(
                    [](const EKF& self) {
                        // return py::make_tuple(self.attr("nStates"), self.attr("dt"), self.attr("xEst"), self.attr("Q"), self.attr("R"), self.attr("P"), self.attr("H"));
                        return py::make_tuple(self.nStates, self.dt, self.getState(), self.getP());
                    },
                    [](py::tuple &t) {
                        if (t.size() != 4)
                            throw std::runtime_error("Invalid state!");

                        EKF ekf(t[0].cast<int>(), t[1].cast<float>(), t[2].cast<state>(), t[3].cast<Eigen::Matrix<float, -1, -1>>());
                        return ekf;
                    }
            ));

    py::class_<Tracker>(m, "Tracker")
            .def(py::init<const obj_m&, const int, const float, const int, const int>())
            .def_readwrite("traj", &Tracker::traj)
            // .def_readwrite("zList", &Tracker::zList)
            // .def_readwrite("predList", &Tracker::predList)
            .def_readwrite("ekf", &Tracker::ekf)
            .def_readwrite("age", &Tracker::age)
            /* .def_readwrite("r", &Tracker::r)
            .def_readwrite("g", &Tracker::g)
            .def_readwrite("b", &Tracker::b) */
            .def_readwrite("cl", &Tracker::cl)
            .def_readwrite("id", &Tracker::id)
            .def_readwrite("idx", &Tracker::idx)
            .def(py::pickle(
                    [](const Tracker& self){
                        return py::make_tuple(self.getTraj(), self.getEKF(), self.age, self.cl, self.id, self.idx);
                    },
                    [](py::tuple &t){
                        std::vector<obj_m> traj = std::move(t[0].cast<std::vector<obj_m>>());
                        // std::vector<state> zList = std::move(t[1].cast<std::vector<state>>());
                        // std::vector<state> predList = std::move(t[2].cast<std::vector<state>>());
                        EKF ekf = std::move(t[1].cast<EKF>());
                        int age = t[2].cast<int>();
                        /* int r = t[5].cast<int>();
                        int g = t[6].cast<int>();
                        int b = t[7].cast<int>(); */
                        int classO = t[3].cast<int>();
                        int id = t[4].cast<int>();
                        int idx = t[5].cast<int>();
                        Tracker tracker(traj, ekf, age, classO, id, idx);
                        return tracker;
                    }
            ));

    m.def("track2", &track2, py::return_value_policy::automatic);
}
