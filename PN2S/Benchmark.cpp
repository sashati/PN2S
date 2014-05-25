//#include "Definitions.h"
#include "core/Solver.h"
#include <cassert>
#include <limits>

#include "Manager.h"

using namespace pn2s;

int main(int argc, char **argv) {

	int size = 3;
	int batch = 10;

	double dt = 1;
	Manager::Setup(dt);

	for (int idx_batch = 0; idx_batch < batch; ++idx_batch) {
		models::Model<CURRENT_TYPE> neutral(idx_batch);

	}

	return 0;
}
