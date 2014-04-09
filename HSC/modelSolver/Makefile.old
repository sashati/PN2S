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

TARGET = _modelSolver.o

COBJS += HSC_SolverChannels.o	\
		HSC_SolverComps.o	\
		HSC_SolverData.o	\
		HSC_Solver.o 		\
		solve.o			\
		
		 
OBJS += HSCModel_HHChannel.o \
	HSCModel.o \
	HSCModel_Compartment.o \
	HSC_NetworkAnalyzer.o \

.PHONY: all
all: $(TARGET)
default: all

HSC_Solver.o: HSC_Solver.cpp HSC_Solver.h HSC_SolverData.h HSC_SolverData.cpp HSC_SolverChannels.h HSC_SolverChannels.cpp HSC_SolverComps.h HSC_SolverComps.cpp 
HSC_SolverData.o: HSC_SolverData.cpp HSC_SolverData.h HSC_SolverChannels.cpp HSC_SolverChannels.h HSC_SolverComps.cpp HSC_SolverComps.h 
HSC_SolverChannels.o: HSC_SolverChannels.cpp HSC_SolverChannels.h 
HSC_SolverComps.o: HSC_SolverComps.cpp HSC_SolverComps.h
HSCModel_HHChannel.o: HSCModel_HHChannel.cpp HSCModel_HHChannel.h
HSCModel.o: HSCModel.cpp HSCModel.h 
HSCModel_Compartment.o: HSCModel_Compartment.cpp HSCModel_Compartment.h
HSC_NetworkAnalyzer.o: HSC_NetworkAnalyzer.cpp HSC_NetworkAnalyzer.h
solve.o: solve.cu solve.h operations.h

$(TARGET): $(OBJS) $(COBJS)
	$(LD) -r -o  $(TARGET) $(OBJS) $(COBJS) 
$(OBJS):
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c 
	
$(COBJS):
	nvcc --compile -o  "$@" "$<" -G -O0 -Xcompiler -fPIC -g -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_20 -x cu  -DDO_UNIT_TESTS

.PHONY: clean
clean:
	rm -rf $(OBJS) $(COBJS) $(TARGET)
