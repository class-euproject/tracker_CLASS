#ifndef EKF_H
#define EKF_H

#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <iostream>

#include <iomanip>

namespace tracking{

struct state
{
  float x          = 0;
  float y          = 0;
  float yaw        = 0;
  float vel        = 0;
  float yawRate   = 0;

  state() {}
  state(const float x_, const float y_, const float yaw_, const float vel_, 
        const float yaw_rate): x(x_), y(y_), yaw(yaw_), vel(vel_), 
        yawRate(yaw_rate) {}
  void print();
};

class EKF
{
public:
  using EKFMatrixF = Eigen::Matrix<float, -1, -1>;
  int nStates;
  float dt;
  state xEst;
  EKFMatrixF Q;
  EKFMatrixF R;
  EKFMatrixF P;
  EKFMatrixF H;

  EKF();
  EKF(const EKF& ekf) : nStates(ekf.nStates), dt(ekf.dt), xEst(ekf.xEst), Q(ekf.Q), R(ekf.R), P(ekf.P), H(ekf.H) {};
  EKF(const int n_states, const float dt_, const EKFMatrixF &Q_, const EKFMatrixF &R_, const state &in_state);
  ~EKF();
  void printInternalState();
  void ekfStep(const EKFMatrixF &H_, const Eigen::VectorXf &z);
  state getEstimatedState();

  EKF& operator=(EKF&& clase){
      this->nStates = clase.nStates;
      this->dt = clase.dt;
      this->xEst = clase.xEst;
      this->Q = std::move(clase.Q);
      this->R = std::move(clase.R);
      this->P = std::move(clase.P);
      this->H = std::move(clase.H);
      return *this;
  }

    EKF& operator=(const EKF& clase){
        this->nStates = clase.nStates;
        this->dt = clase.dt;
        this->xEst = clase.xEst;
        this->Q = clase.Q;
        this->R = clase.R;
        this->P = clase.P;
        this->H = clase.H;
        return *this;
    }

private:
    /*
  int nStates;
  float dt;
  state xEst;
  EKFMatrixF Q;
  EKFMatrixF R;
  EKFMatrixF P;
  EKFMatrixF H;
     */

  state stateTransition();
  EKFMatrixF jacobian(const state &x);
};

}

#endif /*EKF_H*/
