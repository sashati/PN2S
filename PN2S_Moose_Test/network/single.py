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

# Gate equations have the form:
#
# y(x) = (A + B * x) / (C + exp((x + D) / F))
#
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


def create_spine(parentCompt, parentObj, index, frac, length, dia, theta):
    """Create spine of specified dimensions and index"""
    RA = 1.0
    RM = 1.0
    CM = 0.01
    sname = 'shaft' + str(index)
    hname = 'head' + str(index)
    shaft = moose.SymCompartment(parentObj.path + '/' + sname)
    #moose.connect( parentCompt, 'cylinder', shaft, 'proximalOnly','Single')
    #moose.connect( parentCompt, 'distal', shaft, 'proximal','Single' )
    moose.connect(parentCompt, 'sphere', shaft, 'proximalOnly', 'Single')
    x = parentCompt.x0 + frac * (parentCompt.x - parentCompt.x0)
    y = parentCompt.y0 + frac * (parentCompt.y - parentCompt.y0)
    z = parentCompt.z0 + frac * (parentCompt.z - parentCompt.z0)
    shaft.x0 = x
    shaft.y0 = y
    shaft.z0 = z
    sy = y + length * math.cos(theta * math.pi / 180.0)
    sz = z + length * math.sin(theta * math.pi / 180.0)
    shaft.x = x
    shaft.y = sy
    shaft.z = sz
    shaft.diameter = dia / 2.0
    shaft.length = length
    xa = math.pi * dia * dia / 400.0
    circumference = math.pi * dia / 10.0
    shaft.Ra = RA * length / xa
    shaft.Rm = RM / (length * circumference)
    shaft.Cm = CM * length * circumference
    shaft.Em = EREST_ACT
    shaft.initVm = EREST_ACT

    print shaft.Ra
    head = moose.SymCompartment(parentObj.path + '/' + hname)
    moose.connect(shaft, 'distal', head, 'proximal', 'Single')
    head.x0 = x
    head.y0 = sy
    head.z0 = sz
    hy = sy + length * math.cos(theta * math.pi / 180.0)
    hz = sz + length * math.sin(theta * math.pi / 180.0)
    head.x = x
    head.y = hy
    head.z = hz
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


def create_spine_with_receptor(compt, cell, index, frac):
    spineLength = 5.0e-6
    spineDia = 4.0e-6
    head = create_spine(compt, cell, index, frac, spineLength, spineDia, 0.0)
    gluR = moose.SynChan(head.path + '/gluR')
    gluR.tau1 = 4e-3
    gluR.tau2 = 4e-3
    gluR.Gbar = 1e-6
    gluR.Ek =  10.0e-3  # Inhibitory -0.1
    moose.connect(head, 'channel', gluR, 'channel', 'Single')

    synHandler = moose.SimpleSynHandler(head.path + '/gluR/handler')
    synHandler.synapse.num = 1
    moose.connect(synHandler, 'activationOut', gluR, 'activation', 'Single')

    return gluR


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


def make_spiny_compt(root_path, number, synInput):
    comptLength = 100e-6
    comptDia = 4e-6
    numSpines = number_of_spines
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

    for i in range(numSpines):
        r = create_spine_with_receptor(compt, cell, i, i / float(numSpines))
        #r.synapse.num = 1
        syn = moose.element(r.path + '/handler/synapse')
        moose.connect(synInput, 'spikeOut', syn, 'addSpike', 'Single')
        syn.weight = .5
        syn.delay = i * 1.0e-6


def createCells(net):
    network = moose.Neutral(net)
    synInput = moose.SpikeGen("%s/synInput" % net)
    synInput.refractT = 74e-3
    synInput.threshold = -1.0
    synInput.edgeTriggered = False
    synInput.Vm(0)
    for i in range(number_of_ext_cells):
        make_spiny_compt(network.path, i, synInput)


def test_elec_alone():
    moose.setClock(0, dt)
    moose.setClock(1, dt)
    moose.setClock(2, dt)
    moose.setClock(8, 1e-4)

    createCells("/cpu")

    for i in range(number_of_ext_cells):
        hsolve = moose.HSolve('/cpu/cell' + str(i) + '/hsolve')
        hsolve.dt = dt
        hsolve.target = '/cpu/cell' + str(i) + '/compt'

    if Use_MasterHSolve:
        print "*****************"
        createCells("/gpu")
        moose.useClock(0, '/gpu/##[ISA=Compartment]', 'init')
        moose.useClock(1, '/gpu/##', 'process')
        for i in range(number_of_ext_cells):
            hsolve = moose.HSolve('/gpu/cell' + str(i) + '/hsolve')
            hsolve.dt = dt
            moose.useClock(1, '/gpu/cell' + str(i) + '/hsolve', 'process')
            hsolve.target = '/gpu/cell' + str(i) + '/compt'
            
        hsolve = moose.HSolve('/gpu/hsolve')
        hsolve.dt = dt
        moose.useClock(1, '/gpu/hsolve', 'process')
        hsolve.target = '/gpu'

    moose.useClock(0, '/cpu/##', 'init')
    moose.useClock(1, '/cpu/##', 'process')

    moose.Neutral('/graphs')
    moose.Neutral('/graphs/cpu')
    moose.Neutral('/graphs/gpu')

    add_plot("/cpu/cell0" + '/head0', 'getVm', 'cpu/c0_head')
    # add_plot("/cpu/cell0" + '/compt', 'getVm', 'cpu/c0_compt')
    add_plot("/gpu/cell0" + '/head0', 'getVm', 'gpu/c0_head')
    # # add_plot("/cpu/cell0" + '/shaft0', 'getVm', 'cpu/c0_shaft0')
    # add_plot("/gpu/cell0" + '/compt', 'getVm', 'gpu/g0_compt')
    # add_plot("/cpu/synInput", 'getHasFired', 'cpu/sp')
    # add_plot("/cpu/cell0" + '/compt', 'getVm', 'cpu/c0_compt')
    # add_plot("/gpu/cell0" + '/compt', 'getVm', 'gpu/c0_compt')

    # add_plot("/gpu/cell0" + '/head0', 'getVm', 'gpu/c0_comp')
    moose.useClock(8, '/graphs/##', 'process')

    moose.reinit()
    moose.start(Simulation_Time)
    dump_plots()
    pylab.legend()
    pylab.show()


def main():
    test_elec_alone()

Use_MasterHSolve = True
# Use_MasterHSolve = False
Simulation_Time = 1e-2

number_of_input_cells = 1
number_of_ext_cells = 2
number_of_inh_cells = 0
number_of_spines = 2

INJECT_CURRENT = 1e-7
dt = 1e-6

if __name__ == '__main__':
    main()
