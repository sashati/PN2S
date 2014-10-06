import sys
import os
import time

if __name__ == '__main__':
    
    model_size = [100, 500, 1000, 10000]
    fname = 'result.log'
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    start_time = time.time()
    
    for s in model_size:
        os.system("python model.py gpu %d %s" % (s,fname))

    t_exec = time.time() - start_time
    print "\n Simulation time: %s sec\t" % str(t_exec)
