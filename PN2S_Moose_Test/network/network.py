import sys
sys.path.append('../../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import pylab
import numpy
import math
from numpy import random as nprand

import moose
import moose.utils

EREST_ACT = -70e-3

# Gate equations have the form:
#
# y(x) = (A + B * x) / (C + exp((x + D) / F))

# where x is membrane voltage and y is the rate constant for gate
# closing or opening

Na_m_params = [1e5 * (25e-3 + EREST_ACT),   # 'A_A':
               -1e5,                       # 'A_B':
               -1.0,                       # 'A_C':
               -25e-3 - EREST_ACT,         # 'A_D':
               -10e-3,                      # 'A_F':
               4e3,                     # 'B_A':
               0.0,                        # 'B_B':
               0.0,                        # 'B_C':
               0.0 - EREST_ACT,            # 'B_D':
               18e-3                       # 'B_F':
               ]
Na_h_params = [70.0,                        # 'A_A':
               0.0,                       # 'A_B':
               0.0,                       # 'A_C':
               0.0 - EREST_ACT,           # 'A_D':
               0.02,                     # 'A_F':
               1000.0,                       # 'B_A':
               0.0,                       # 'B_B':
               1.0,                       # 'B_C':
               -30e-3 - EREST_ACT,        # 'B_D':
               -0.01                    # 'B_F':
               ]
K_n_params = [1e4 * (10e-3 + EREST_ACT),  # 'A_A':
              -1e4,  # 'A_B':
              -1.0,  # 'A_C':
              -10e-3 - EREST_ACT,  # 'A_D':
              -10e-3,  # 'A_F':
              0.125e3,  # 'B_A':
              0.0,  # 'B_B':
              0.0,  # 'B_C':
              0.0 - EREST_ACT,  # 'B_D':
              80e-3  # 'B_F':
              ]
VMIN = -30e-3 + EREST_ACT
VMAX = 120e-3 + EREST_ACT
VDIVS = 3000


def create_squid(parent):
    """Create a single compartment squid model."""
    compt = moose.SymCompartment(parent.path + '/compt')
    Em = EREST_ACT + 10.613e-3
    compt.Em = Em
    compt.initVm = EREST_ACT
    compt.Cm = 7.85e-9 * 0.5
    compt.Rm = 4.2e5 * 5.0
    compt.Ra = 7639.44e3

    nachan = moose.HHChannel(parent.path + '/compt/Na')
    nachan.Xpower = 3
    xGate = moose.HHGate(nachan.path + '/gateX')
    xGate.setupAlpha(Na_m_params + [VDIVS, VMIN, VMAX])
    # This is important: one can run without it but the output will diverge.
    xGate.useInterpolation = 1
    nachan.Ypower = 1
    yGate = moose.HHGate(nachan.path + '/gateY')
    yGate.setupAlpha(Na_h_params + [VDIVS, VMIN, VMAX])
    yGate.useInterpolation = 1
    nachan.Gbar = 0.942e-3
    nachan.Ek = 115e-3 + EREST_ACT
    moose.connect(nachan, 'channel', compt, 'channel', 'OneToOne')
    kchan = moose.HHChannel(parent.path + '/compt/K')
    kchan.Xpower = 4.0
    xGate = moose.HHGate(kchan.path + '/gateX')
    xGate.setupAlpha(K_n_params + [VDIVS, VMIN, VMAX])
    xGate.useInterpolation = 1
    kchan.Gbar = 0.2836e-3
    kchan.Ek = -12e-3 + EREST_ACT
    moose.connect(kchan, 'channel', compt, 'channel', 'OneToOne')
    return compt


def create_dendrit(parentCompt, parentObj, index, frac, length, dia, theta):
    """Create spine of specified dimensions and index"""
    RA = 1.0
    RM = 1.0
    CM = 0.01
    hname = 'head' + str(index)
    head = moose.SymCompartment(parentObj.path + '/' + hname)
#     moose.connect(parentCompt, 'distal', head, 'proximal', 'Single')
    head.diameter = dia
    head.length = length
    xa = math.pi * dia * dia / 4.0
    circumference = math.pi * dia
    head.Ra = RA * length / xa
    head.Rm = RM / (length * circumference)
    head.Cm = CM * length * circumference
    head.Em = EREST_ACT
    head.initVm = EREST_ACT
    return head


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
    # for x in moose.wildcardFind('/graphs/gpu/##[ISA=Table]'):
    #     t = numpy.arange(0, len(x.vector), 1)
    #     pylab.plot(t, x.vector, label=("GPU:%s" % x.name))


def make_spiny_compt(root_path, number):
    comptLength = 100e-6
    comptDia = 4e-6
    cell = moose.Neutral(root_path + "/cell" + str(number))

    compt = create_squid(cell)
    compt.inject = 0
    compt.x0 = 0
    compt.y0 = 0
    compt.z0 = 0
    compt.x = comptLength
    compt.y = 0
    compt.z = 0
    compt.length = comptLength
    compt.diameter = comptDia

    #Excitatory
    spineLength = 5.0e-6
    spineDia = 4.0e-6
    head = create_dendrit(compt, cell, 0, 0, spineLength, spineDia, 0.0)
    gluR = moose.SynChan(head.path + '/gluR')
    gluR.tau1 = 4e-3
    gluR.tau2 = 4e-3
    gluR.Gbar = 1e-6
    gluR.Ek = 10.0e-3  # Inhibitory -0.1
    moose.connect(head, 'channel', gluR, 'channel', 'Single')

    synHandler = moose.SimpleSynHandler(head.path + '/gluR/handler')
    synHandler.synapse.num = 1
    moose.connect(synHandler, 'activationOut', gluR, 'activation', 'Single')
    
    
    return gluR
#     syn = moose.element(r.path + '/handler/synapse')
#     moose.connect(synInput, 'spikeOut', syn, 'addSpike', 'Single')
#     syn.weight = .5
#     syn.delay = 0
        
#     #Inhibitory
#     r = create_spine_with_receptor(compt, cell, 1, 0.5)
#     syn = moose.element(r.path + '/handler/synapse')
#     moose.connect(synInput, 'spikeOut', syn, 'addSpike', 'Single')
#     syn.weight = .5
#     syn.delay = 0 * 1.0e-6 #??
        
        

def createCells(net, input_layer):
    network = moose.Neutral(net)
    
    for i in range(number_of_ext_cells):
        r = make_spiny_compt(network.path, i)
#         rnd = nprand.rand(1)
        
        for j in range(number_of_input_cells):
#         if( rnd >  IC):
            syn = moose.element(r.path + '/handler/synapse')
#             index = nprand.randint(number_of_input_cells)
            moose.connect(input_layer[j], 'spikeOut', syn, 'addSpike', 'Single')
            syn.weight = .1
            syn.delay = 5e-3 #random
        


def make_input_layer(avgFiringRate=10, spike_refractT=74e-4):
    _input = moose.Neutral('/in')

    stim = moose.StimulusTable(_input.path + '/stim')
    stim.vector = [avgFiringRate]
    stim.startTime = 0
    stim.stopTime = 1
    stim.loopTime = 1
    stim.stepSize = 0
    stim.stepPosition = 0
    stim.doLoop = 1

    input_layer = []
    for i in range(number_of_input_cells):
        spike = moose.RandSpike(_input.path + '/sp'+ str(i))
        spike.refractT = spike_refractT
        moose.connect(stim, 'output', spike, 'setRate')
        input_layer.append(spike)

    return input_layer


def main():
    moose.Neutral('/graphs')
    moose.Neutral('/graphs/cpu')
    moose.Neutral('/graphs/gpu')

    moose.setClock(0, dt)
    moose.setClock(1, dt)
    moose.setClock(2, dt)
    moose.setClock(8, 1e-4)

    input_layer = make_input_layer()

    createCells("/cpu", input_layer)

    for i in range(number_of_ext_cells):
        add_plot("/cpu/cell" + str(i) + '/head0', 'getVm', 'cpu/c'+ str(i)+'_head')
        add_plot("/cpu/cell" + str(i) + '/compt', 'getVm', 'cpu/c'+ str(i)+'_cmpt')
    
#     add_plot("/cpu/stim" , 'getOutputValue', 'cpu/stim')
    

    moose.useClock(0, '/cpu/##', 'init')
    moose.useClock(1, '/cpu/##', 'process')
    moose.useClock(1, '/in/stim', 'process')
    moose.useClock(1, '/in/sp#', 'process')    
    moose.useClock(8, '/graphs/##', 'process')
    moose.reinit()
    moose.start(Simulation_Time)
    dump_plots()
    pylab.legend()
    pylab.show()


# Use_MasterHSolve = True
Use_MasterHSolve = False
Simulation_Time = 1

number_of_input_cells = 1
number_of_ext_cells = 1
number_of_inh_cells = 2


IC = .5 #Input connection probability


INJECT_CURRENT = 1e-6
dt = 1e-6

if __name__ == '__main__':
    main()
