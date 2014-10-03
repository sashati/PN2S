import getopt
import subprocess
import sys
import os
import socket
import multiprocessing
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
import random
from fileinput import filename


if __name__ == '__main__':
    
    stream_numbers = [8, 4, 2, 1]
    model_pack_size = [32, 64, 128, 256, 512, 800, 1024, 2048, 3072, 4086, 5120, 6144, 7168, 8192, 9216, 10240, 20480, 30720]
    model_size = [10, 50, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 4000, 8000, 10000]
    fname = 'result.log'
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    start_time = datetime.now()
#     Header
    with open(fname, 'a') as f:
        f.write("\n\n\n")    
        for sn in stream_numbers:
            for m in model_pack_size:
                f.write("\t(%d,%d)" % (sn,m))
        f.flush()
#     for s in model_size:
#         execfile("model.py %s" + str())
#         te = os.environ['EXEC_TIME']
#         ti = os.environ['INIT_TIME']
    
    for s in model_size:
        
        with open(fname, 'a') as f:
            f.write("\n %d \t" % s)
            f.flush()
             
        for sn in stream_numbers:
            for m in model_pack_size:
                os.system("python model.py gpu %d %d %d %s" % (s,m,sn,fname))

    with open(fname, 'a') as f:
        f.write("\n Simulation time: %d sec\t" % datetime.now - start_time)
        f.flush()
        