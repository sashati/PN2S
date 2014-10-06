import sys
import os
import time

if __name__ == '__main__':
    
    model_size = 10
    fname = 'result-1000.log'
    
    os.system("python model3.py gpu %d %s" % (model_size,fname))
    os.system("python model4.py gpu %d %s" % (model_size,fname))
    os.system("python model5.py gpu %d %s" % (model_size,fname))
    os.system("python model6.py gpu %d %s" % (model_size,fname))
    
