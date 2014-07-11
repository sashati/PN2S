#pragma once

namespace pn2s {
namespace models {

class ModelStatistic {
public:
	double dt;
	size_t nModels;
	size_t nCompts;
	size_t nChannels;
	ModelStatistic(): dt(1), nModels(0), nCompts(0), nChannels(0) {}
	ModelStatistic(double _dt, size_t _nModel, size_t _nCompt, size_t numberOfChannels):
		dt(_dt), nModels(_nModel), nCompts(_nCompt), nChannels(numberOfChannels){}
	virtual ~ModelStatistic(){}
};

}
}
