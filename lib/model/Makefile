PYTHON_INC = /users/gpu/samiks/anaconda3/envs/pytorch-cuda10/include/python3.7m
PYTHON_LIB = /users/gpu/samiks/anaconda3/envs/pytorch-cuda10/lib/python3.7/config-3.7m-x86_64-linux-gnu
BOOST_INC = /users/gpu/samiks/anaconda3/envs/pytorch-cuda10/include
BOOST_LIB = /users/gpu/samiks/anaconda3/envs/pytorch-cuda10/lib
EIGEN_INC = /users/gpu/samiks/anaconda3/envs/pytorch-cuda10/include/eigen3

TARGET = _CppDPP
$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python37 -lboost_numpy37 -L$(PYTHON_LIB) -lpython3.7m -fopenmp -o $(TARGET).so
$(TARGET).o: $(TARGET).cpp
	g++ -O3 -march=native -fopenmp -I$(PYTHON_INC) -I$(BOOST_INC) -I$(EIGEN_INC) -fPIC -c $(TARGET).cpp
