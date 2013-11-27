#include "header.h"
#include "HinesMatrixProxy.h"

#include "cudaLibrary/HSC_HinesMatrix.hpp"
#include "cudaLibrary/PlatformFunctions.hpp"

//#include "HinesStruct.hpp"
#include "cudaLibrary/Connections.hpp"
#include "cudaLibrary/SpikeStatistics.hpp"
#include "cudaLibrary/NeuronInfoWriter.hpp"


#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>

#ifndef PERFORMSIMULATION_H_
#define PERFORMSIMULATION_H_

class HSC_PerformSimulation {

private:
	struct ThreadInfo * tInfo;
	struct SharedNeuronGpuData *sharedData;
	struct KernelInfo *kernelInfo;

public:
    HSC_PerformSimulation(struct ThreadInfo *tInfo);
    int launchExecution();
    int setup(const vector< TreeNodeStruct >& tree, double dt);
    int performHostExecution();
private:
    void addReceivedSpikesToTargetChannelCPU();
    void syncCpuThreads();
    void updateBenchmark();
    void createNeurons( ftype dt );
    void createActivationLists( );
    void initializeThreadInformation();
    void updateGenSpkStatistics(int *& nNeurons, struct SynapticData *& synData);
    void generateRandomSpikes(int type, struct RandomSpikeInfo & randomSpkInfo);
};

#endif /* PERFORMSIMULATION_H_ */

