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
    
    model_pack_size = [2048, 10240, 30720]
    model_size = [10, 50, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 4000, 8000]
    fname = 'result.log'
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    start_time = datetime.now()
    
    for s in model_size:
        for m in model_pack_size:
            os.system("python model.py gpu %d %d %s" % (s,m,fname))

    print "\n Simulation time: %d sec\t" % datetime.now - start_time
