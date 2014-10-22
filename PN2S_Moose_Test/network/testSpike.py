import sys
sys.path.append('../../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import pylab
import numpy
import math

import moose
import moose.utils

EREST_ACT = -70e-3



def add_plot(objpath, field, plot):
    assert moose.exists(objpath)
    tab = moose.Table('/graphs/' + plot)
    obj = moose.element(objpath)
    moose.connect(tab, 'requestOut', obj, field)
    return tab


def dump_plots():
    # if ( os.path.exists( fname ) ):
    #     os.remove( fname )
    for x in moose.wildcardFind('/graphs/cpu/##[ISA=Table]'):
        # print x.vector
        t = numpy.arange(0, len(x.vector), 1)
        pylab.plot(t, x.vector, label=("CPU:%s" % x.name))
    for x in moose.wildcardFind('/graphs/gpu/##[ISA=Table]'):
        t = numpy.arange(0, len(x.vector), 1)
        pylab.plot(t, x.vector, label=("GPU:%s" % x.name))


def createCells(net):
    network = moose.Neutral(net)
    comptLength = 100e-6
    comptDia = 4e-6
    cell = moose.Neutral(network.path+"/cell")

    compt = moose.SymCompartment(cell.path + '/compt')
    compt.Em = EREST_ACT + 10.613e-3
    compt.initVm = EREST_ACT
    compt.Cm = 7.85e-9 * 0.5
    compt.Rm = 4.2e5 * 5.0
    compt.Ra = 7639.44e3

    compt.inject = 0
    compt.x0 = 0
    compt.y0 = 0
    compt.z0 = 0
    compt.x = comptLength
    compt.y = 0
    compt.z = 0
    compt.length = comptLength
    compt.diameter = comptDia

    gluR = moose.SynChan(compt.path + '/gluR')
    gluR.tau1 = 4e-3
    gluR.tau2 = 4e-3
    gluR.Gbar = 1e-6
    gluR.Ek = 10.0e-3  # Inhibitory -0.1
    gluR.synapse.num = 1
    moose.connect(compt, 'channel', gluR, 'channel', 'Single')

    syn = moose.element(gluR.path + '/synapse')

    synInput = moose.SpikeGen("%s/synInput" % network.path)
    synInput.refractT = 20e-3
    synInput.threshold = -1.0
    synInput.edgeTriggered = False
    synInput.Vm(0)

    moose.connect(synInput, 'spikeOut', syn, 'addSpike', 'Single')
    syn.weight = .5
    syn.delay = 1.0e-6


def test_elec_alone():
    moose.setClock(0, dt)
    moose.setClock(1, dt)
    moose.setClock(2, dt)
    moose.setClock(8, 1e-4)
    createCells("/cpu")

    moose.useClock(0, '/cpu/##', 'init')
    moose.useClock(1, '/cpu/##', 'process')

    moose.Neutral('/graphs')
    moose.Neutral('/graphs/cpu')
    moose.Neutral('/graphs/gpu')

    # add_plot('/cpu/cell0/compt/gluR', 'getGk', 'cpu/c0_head')
    # add_plot("/cpu/synInput", 'getHasFired', 'cpu/sp')
    add_plot("/cpu/cell" + '/compt', 'getVm', 'cpu/c0_compt')
    # add_plot("/gpu/cell0" + '/compt', 'getVm', 'gpu/c0_compt')

    # add_plot("/gpu/cell0" + '/head0', 'getVm', 'gpu/c0_comp')
    moose.useClock(1, '/graphs/##', 'process')

    moose.reinit()
    moose.start(Simulation_Time)
    dump_plots()
    pylab.legend()
    pylab.show()


def run_simulator():
    test_elec_alone()

# Use_MasterHSolve = True
Use_MasterHSolve = False
Simulation_Time = 1e-1

number_of_input_cells = 1
number_of_ext_cells = 1
number_of_inh_cells = 0
number_of_spines = 1

INJECT_CURRENT = 1e-7
dt = 1e-6

if __name__ == '__main__':
    run_simulator()
