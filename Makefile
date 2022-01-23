CXX := g++
INC_DIRS := $(shell find src -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
CFLAGS := -fPIC -std=c++11 ${INC_FLAGS} -g
PYBIND_INC := $(shell python3 -m pybind11 --includes)

SUFFIX := $(shell python3-config --extension-suffix)

build/_tinydl${SUFFIX}: build/tensor.o build/opr/elementwise.o build/opr/reduction.o build/opr/matmul.o build/tinydl.o build/autograd.o
	${CXX} $^ -o $@ ${PYBIND_INC} ${CXXFLAG} -shared -lcuda -lcudart
	cp $@ tinydl/_tinydl${SUFFIX}



build/tensor.o: src/tensor.cpp src/tensor.h build
	${CXX} ${CFLAGS} ${PYBIND_INC} -c -g $< -o $@

build/autograd.o: src/autograd.cpp src/autograd.h src/tensor.h src/opr/elementwise.h build
	${CXX} ${CFLAGS} ${PYBIND_INC} -c -g $< -o $@

build/opr/elementwise.o: src/opr/elementwise.cu src/opr/elementwise.h src/tensor.h src/autograd.h
	nvcc -std=c++11 ${PYBIND_INC} ${INC_FLAGS} -g -Xcompiler="-fPIC" -c $< -o $@ 

build/opr/reduction.o: src/opr/reduction.cu build
	nvcc -std=c++11 ${PYBIND_INC} ${INC_FLAGS} -g -Xcompiler="-fPIC" -c $< -o $@ 

build/opr/matmul.o: src/opr/matmul.cu build
	nvcc -std=c++11 ${PYBIND_INC} ${INC_FLAGS} -g -Xcompiler="-fPIC" -c $< -o $@ 


build/tinydl.o: src/tinydl.cpp build
	nvcc -std=c++11 ${PYBIND_INC} ${INC_FLAGS} -g -Xcompiler="-fPIC" -c $< -o $@ 

build:
	mkdir $@
	mkdir build/opr

clean:
	-rm build/**/*.o build/*.so
