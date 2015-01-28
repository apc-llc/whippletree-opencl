AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16
all: queue

cfiles.o: ../../segmentedStorage.c
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c

queue.o: main.cpp 
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c

queue: queue.o cfiles.o
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL

clean:
	rm -rf *.o queue

