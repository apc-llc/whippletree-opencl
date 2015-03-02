AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16
all: clean queue

compile.o:../../comp.c
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c -g

cfiles.o: ../../segmentedStorage.c
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c -g

test.o: test.cpp
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c -g

queue.o: main.cpp
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c -g

queue: queue.o test.o cfiles.o compile.o
	g++ -std=c++11 -I $(AMD)include -I ../.. -L$(AMD)lib/x86_64 $^ -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -g

clean:
	rm -rf *.o queue

