AMD=/opt/AMDAPPSDK-3.0-0-Beta/
BLOCK_SIZE=16
all: comp

comp.o: comp.c
	g++ -I $(AMD)include -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL -DBLOCK_SIZE=$(BLOCK_SIZE) -c

comp: comp.o
	g++ -I $(AMD)include -L$(AMD)lib/x86_64 $< -o $@ -Wl,-rpath,$(AMD)lib/x86_64 -lOpenCL

clean:
	rm -rf *.o comp

