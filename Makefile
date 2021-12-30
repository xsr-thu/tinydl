CXX := g++
CFLAGS := -fPIC -std=c++11
PYBIND_INC := $(shell python3 -m pybind11 --includes)

SUFFIX := $(shell python3-config --extension-suffix)

build/tinydl${SUFFIX}: build/tensor.o build/elementwise.o build/reduction.o build/tinydl.o
	${CXX} $^ -o $@ ${PYBIND_INC} ${CXXFLAG} -shared -lcuda -lcudart


build/tensor.o: src/tensor.cpp src/tensor.h build
	${CXX} ${CFLAGS} ${PYBIND_INC} -c $< -o $@

build/elementwise.o: src/elementwise.cu src/elementwise.h build
	nvcc -std=c++11 ${PYBIND_INC} -Xcompiler="-fPIC" -c $< -o $@ 

build/reduction.o: src/reduction.cu src/reduction.h build
	nvcc -std=c++11 ${PYBIND_INC} -Xcompiler="-fPIC" -c $< -o $@ 

build/tinydl.o: src/tinydl.cpp build
	nvcc -std=c++11 ${PYBIND_INC} -Xcompiler="-fPIC" -c $< -o $@ 

build:
	mkdir $@

clean:
	-rm build/*.o build/*.so
