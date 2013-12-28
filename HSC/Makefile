#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
#TODO: imported from main Makefile for test, after we can remove it
BUILD=debug
ifeq ($(BUILD),debug)
CXXFLAGS = -g -pthread -fpermissive -fno-strict-aliasing -fPIC -fno-inline-functions -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_GSL = 1
endif
####################################################################

TARGET = _hsc.o

#COBJS += HSC_Device.o \
		 
OBJS += HSC_Manager.o \
	HSC_Scheduler.o \

SUBDIR = \
	modelSolver \
	
MODULES = modelSolver/_modelSolver.o

.PHONY: all
all: $(TARGET)
default: all

HSC_Manager.o: HSC_Manager.cpp HSC_Manager.h 
HSC_Scheduler.o: HSC_Scheduler.cpp HSC_Scheduler.h 
HSC_Device.o: HSC_Device.cpp HSC_Device.h 

$(TARGET): $(OBJS) $(COBJS) $(SUBDIR)
#	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i; done)
	$(LD) -r -o  $(TARGET) $(OBJS) $(COBJS) $(MODULES)
	 
$(OBJS):
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c 
	
$(COBJS):
	nvcc --compile -o  "$@" "$<" -G -O0 -Xcompiler -fPIC -g -gencode arch=compute_30,code=sm_30  -x cu     

.PHONY: clean
$(SUBDIR): force
	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i; done)

.PHONY: force
force :;

test: $(TARGET) 
	$(CXX)  HSC_Test.cpp $(TARGET) -o HSC_Test -L/usr/local/cuda/lib64 -lcuda -lcudart $(CXXFLAGS)

clean:
	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i clean;  done)
	rm -rf $(OBJS) $(COBJS) $(TARGET) HSC_Test

#TARGET = _hsc.o
#
#COBJS += HSC_Solver.o 		\
#		 
#OBJS += 
#		
#TEST_TARGET = HSC_Test
#TEST_OBJ = HSC_Test.o
##PROFILE_TARGET = HSC_prof.out
#CXXFLAGS = -g -O0
#
##HEADERS = \
#	../basecode/header.h
#

#
#default: $(TARGET)
#	/usr/local/cuda-5.5/bin/nvcc -ccbin g++ -G -O0 -g -I./inc  -m64    -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\" -o simpleStreams simpleStreams.cu
#	#/usr/local/cuda-5.5/bin/nvcc -I"/usr/local/cuda-5.5/samples/0_Simple" -I"/usr/local/cuda-5.5/samples/common/inc" -I"/src/saeed/cuda-workspace/streams" -G -g -O0 -m64 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "src/simpleStreams.d" "../src/simpleStreams.cu"
# 	#/usr/local/cuda-5.5/bin/nvcc --compile -G -I"/usr/local/cuda-5.5/samples/0_Simple" -I"/usr/local/cuda-5.5/samples/common/inc" -I"/src/saeed/cuda-workspace/streams" -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -m64  -x cu -o  "src/simpleStreams.o" "../src/simpleStreams.cu"
#	
#	
#$(OBJ)	: $(HEADERS)
#
#.cpp.o:
#	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c -I/usr/local/cuda/include
#
#$(TARGET): $(SUBDIR) $(TEST_OBJ) $(OBJ) $(SMOLDYN_OBJ) $(HEADERS)  
#	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS) $(CLIB)
#	$(CXX) $(CXXFLAGS) $(TEST_OBJ) $(TARGET) -o $(TEST_TARGET)
#	

#	
##	valgrind --tool=callgrind ./HSC_test
##	kcachegrind $(ls "callgrind.out.*")
#	
#clean:
#	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i clean;  done)
#	rm -f *.o $(TARGET) $(TEST_TARGET) $(PROFILE_TARGET) gmon.out callgrind.out*
