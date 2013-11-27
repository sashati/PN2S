#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
#TODO: imported from main Makefile for test, after we can remove it
BUILD=debug
ifeq ($(BUILD),debug)
CXXFLAGS = -g -pthread -fpermissive -fno-strict-aliasing -fPIC -fno-inline-functions -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -I/usr/local/cuda/include
USE_GSL = 1
endif
####################################################################


OBJS =	\
	PerformSimulation.o \
 	
CLIB = ../cudaLibrary/_hsolveCudaLib.o
TARGET = testHines

default: $(TARGET)

$(TARGET):	$(OBJS)
	g++ -o $(TARGET) $(OBJS) $(CLIB) launchHines.cpp -L/usr/local/cuda/lib64 -lcuda -lcudart -lm  -lpthread 

PerformSimulation.o:	PerformSimulation.cpp	PerformSimulation.hpp

$(OBJS):
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c
	
clean:
	rm -f $(OBJS) $(TARGET) launchHines.o 
