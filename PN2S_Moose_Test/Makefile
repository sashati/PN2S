CXXFLAGS =	-g -Wall -fmessage-length=0 -lm -I/usr/include/python2.7

OBJS =		PN2S_Moose_Test.o

LIBS =

TARGET =	PN2S_Moose_Test

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS) -lm -lpython2.7 -L/usr/lib/python2.7/config

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) *.plot

test_performance:
	 nvcc -G -O0 -Xcompiler -fopenmp -Xcompiler -fPIC -g -gencode arch=compute_30,code=compute_30 -I./libs -I./libs/tbb/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lm -lgomp  -o test_perf test_perf.cu ../PN2S/_pn2s.o
	