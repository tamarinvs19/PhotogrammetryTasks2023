#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    std::pair<int, int> size = {2*count, 3};
    Eigen::MatrixXd A(size.first, size.second);
    Eigen::VectorXd b(size.first);

    for (int i_pair = 0; i_pair < count; ++i_pair) {
        double x = ms[i_pair][0];
        double y = ms[i_pair][1];
        double z = ms[i_pair][2];

        for (int i = 0; i < 3; i++) {
            A(2*i_pair, i) = x * Ps[i_pair](2, i) - z * Ps[i_pair](0, i);
            A(2*i_pair+1, i) = y * Ps[i_pair](2, i) - z * Ps[i_pair](1, i);
        }

        b(2*i_pair) = -x * Ps[i_pair](2, 3) + z * Ps[i_pair](0, 3);
        b(2*i_pair+1) = -y * Ps[i_pair](2, 3) + z * Ps[i_pair](1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd s_inv = Eigen::MatrixXd::Zero(size.second, size.first);
    for (int i = 0; i < 3; i++) {
        s_inv(i, i) = 1 / svda.singularValues()[i];
    }
    Eigen::VectorXd solution = svda.matrixV() * s_inv * svda.matrixU().transpose() * b;
    cv::Vec4d solution_cv = {solution[0], solution[1], solution[2], 1};
    return solution_cv;
}
