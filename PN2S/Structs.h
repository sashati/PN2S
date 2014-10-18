#pragma once


//	Architectures
#define ARCH_SM13       (0)
#define ARCH_SM20       (1)
#define ARCH_SM30       (2)
#define ARCH_SM35       (3)

#define hscID_t uint

namespace pn2s{
//Setter and Getter functions
struct FIELD{
	enum  CM {CM_BY_DT, EM_BY_RM, RA,INIT_VM, VM, INJECT_BASAL, INJECT_VARYING, CONSTANT, EXT_CURRENT_GK,EXT_CURRENT_EKGK};
	enum  GATE { GATE_STATE, GATE_POWER, GATE_CH_GBAR, GATE_CH_GK, GATE_CH_EK, GATE_COMPONENT_INDEX, GATE_CHANNEL_INDEX, GATE_INDEX};
	enum  CH {	CH_X, CH_Y, CH_Z, CH_X_POWER, CH_Y_POWER, CH_Z_POWER, CH_GBAR, CH_GK, CH_EK, CH_COMPONENT_INDEX};
};

struct Location{
	//TODO: Optimize it with assign a 64 bit variable and keep everything in one place
	int16_t device;
	int16_t pack;
	int32_t index;

	Location():device(0), pack(0), index(0){}
	Location(int16_t _d): device(_d), pack(0), index(0){}
	Location(int16_t _d, int16_t _p): device(_d), pack(_p), index(0){}
	Location(int16_t _d,int16_t _p,int32_t _i): device(_d), pack(_p), index(_i){}
	bool operator<(const Location &r)  const {
		if(device < r.device)
			return true;
		int64_t lv = (((int64_t)pack) << 32) | index;
	    int64_t rv = (((int64_t)r.pack) << 32) | r.index;
	    return lv < rv;
	}
};

struct Model_info{
	unsigned int id;
	unsigned int nCompt;
	unsigned int nChannel;
	unsigned int nGates;
	Model_info():id(0), nCompt(0), nChannel(0), nGates(0){}
	Model_info(unsigned int i):id(i), nCompt(0), nChannel(0), nGates(0){}
	Model_info(unsigned int i, unsigned int n, unsigned int c,  unsigned int g):id(i), nCompt(n), nChannel(c), nGates(g){}
	Model_info(const Model_info& m) : id(m.id), nCompt(m.nCompt), nChannel(m.nChannel), nGates(m.nGates) {}
};

typedef vector<Model_info> Model_pack_info;
}
