#include <stdio.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <random>

namespace p = boost::python;
namespace np = boost::python::numpy;
namespace E = Eigen;

void copy_y(int *y, int *y_tmp, int y_n) {
    for (int i = 0; i < y_n; i++) y_tmp[i] = y[i];
}

void insert_y(int *y, int y_n, int e) {
    int i = y_n;
    while (i > 0 && y[i - 1] > e) {
        y[i] = y[i - 1];
        i--;
    }
    y[i] = e;
}

double obj(double *L_ptr, int n, int *y, int y_n) {
    if (y_n == 0) {
        return 0;
    }

    E::MatrixXd L_y(y_n, y_n);
    for (int i = 0; i < y_n; i++) {
        for (int j = 0; j < y_n; j++) {
            L_y(i, j) = L_ptr[y[i] * n + y[j]];
        }
    }

    E::LLT<E::MatrixXd> L_y_llt(L_y);
    auto &llt_l = L_y_llt.matrixL();
    double log_det = 0;
    for (int i = 0; i < y_n; i++) {
        log_det += log(llt_l(i, i));
    }
    log_det *= 2;

    return log_det;
}

struct Element {
    int index;
    double value;
};

void heapify(Element *Y, int Y_n, int i) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    int largest = i;

    if (left < Y_n && Y[left].value > Y[largest].value) {
        largest = left;
    }
    if (right < Y_n && Y[right].value > Y[largest].value) {
        largest = right;
    }

    if (largest != i) {
        int tmp_index = Y[i].index;
        double tmp_value = Y[i].value;
        Y[i].index = Y[largest].index;
        Y[i].value = Y[largest].value;
        Y[largest].index = tmp_index;
        Y[largest].value = tmp_value;

        heapify(Y, Y_n, largest);
    }
}

void build_heap(Element *Y, int Y_n) {
    for (int i = Y_n / 2 - 1; i >= 0; i--) {
        heapify(Y, Y_n, i);
    }
}

int greedy_accelerated(np::ndarray L, int k, np::ndarray ret_y) {
    double *L_ptr = reinterpret_cast<double *>(L.get_data());
    int n = L.shape(0);

    Element *Y = (Element *)malloc(n * sizeof(Element));
    int Y_n = n;
    int *y = (int *)malloc(k * sizeof(int));
    int *y_tmp = (int *)malloc(k * sizeof(int));
    int y_n = 0;

    for (int i = 0; i < n; i++) {
        Y[i].index = i;
        y_tmp[0] = i;
        Y[i].value = obj(L_ptr, n, y_tmp, 1);
    }
    build_heap(Y, Y_n);

    while (Y_n > 0) {
        double curr_val = obj(L_ptr, n, y, y_n);

        int last_idx = -1;

        while (Y[0].index != last_idx) {
            copy_y(y, y_tmp, y_n);
            insert_y(y_tmp, y_n, Y[0].index);
            double new_val = obj(L_ptr, n, y_tmp, y_n + 1);
            Y[0].value = new_val - curr_val;
            last_idx = Y[0].index;
            heapify(Y, Y_n, 0);
        }

        if (Y[0].value > 0) {
            insert_y(y, y_n, Y[0].index);
            y_n++;
            Y[0] = Y[Y_n - 1];
            Y_n--;
            heapify(Y, Y_n, 0);

            if (y_n >= k) {
                break;
            }
        } else {
            break;
        }
    }

    int64_t *ret_y_ptr = reinterpret_cast<int64_t *>(ret_y.get_data());
    for (int i = 0; i < y_n; i++) {
        ret_y_ptr[i] = y[i];
    }

    free(Y);
    free(y);
    free(y_tmp);

    return y_n;
}

BOOST_PYTHON_MODULE(_CppDPP) {
    E::setNbThreads(4);
    np::initialize();
    p::def("greedy", greedy_accelerated);
}