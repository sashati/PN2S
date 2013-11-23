#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
 
OBJS =	\
 	launchHines.o \
 	
CLIB = ../cudaLibrary/_hsolveCudaLib.o
TARGET = testHines

$(TARGET):	$(OBJS)
		g++ -o $(TARGET) $(OBJS) $(CLIB) -L/usr/local/cuda/lib64 -lcuda -lcudart -lm  -lpthread 

default: $(TARGET)

%.o: %.cpp
	g++ $(CXXFLAGS) -c -o "$@" "$<"
	
clean:
	rm -f $(OBJS) $(TARGET)
