AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16

INCLUDE_FLAGS=-I $(AMD)include -I ../..
LD_FLAGS=-L$(AMD)lib/x86_64 -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL 
DEFINE_FLAGS=-DBLOCK_SIZE=$(BLOCK_SIZE)
CXX_FLAGS=-std=c++11 -g
CXX=g++

all: clean tasman cfiles.o DynamicTaskManager.o DynamicTaskManagerInternal.o


cfiles.o: ../../segmentedStorage.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

#test.o: test.cpp
#	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

DynamicTaskManager.o: DynamicTaskManager.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c

matmul.o: matmul.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c
	
DynamicTaskManagerInternal.o: DynamicTaskManagerInternal.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS) -c	

tasman: DynamicTaskManagerInternal.o matmul.o DynamicTaskManager.o cfiles.o
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS)


#queue: queue.o test.o cfiles.o
#	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $^ -o $@ $(LD_FLAGS) $(DEFINE_FLAGS)

clean:
	rm -rf *.o tasman

