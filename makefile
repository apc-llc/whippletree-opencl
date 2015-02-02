AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16
all: comp cppcode.o

comp.o: comp.c
	g++ -I $(AMD)include -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c

comp: comp.o
	g++ -I $(AMD)include -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL

cppcode.o: techniqueMegakernel.cpp
	 g++ -I $(AMD)include -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c
clean:
	rm -rf *.o comp

