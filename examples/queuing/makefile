AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16

INCLUDE_FLAGS=-I $(AMD)include -I ../..
LD_FLAGS=-L$(AMD)lib/x86_64 -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL 
DEFINE_FLAGS=-DBLOCK_SIZE=$(BLOCK_SIZE)
CXX_FLAGS=-std=c++11 -g
CXX=g++

all: clean queue

compile.o: comp.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

cfiles.o: ../../segmentedStorage.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

test.o: test.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

queue.o: main.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

queue: queue.o test.o cfiles.o compile.o
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS)

clean:
	rm -rf *.o queue

