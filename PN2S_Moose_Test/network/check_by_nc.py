import sys
import os
import time

if __name__ == '__main__':

    model_size = [128, 256, 512, 1024]
    fname = 'result.log'

    if len(sys.argv) > 1:
        fname = sys.argv[1]

    for s in model_size:
    	os.system("python model3.py cpu %d %s" % (s,fname))
    	os.system("python model4.py cpu %d %s" % (s,fname))
    	os.system("python model5.py cpu %d %s" % (s,fname))
    	os.system("python model6.py cpu %d %s" % (s,fname))

