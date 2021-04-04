import nvsmi
import numpy as np

def getFreeGpu():
    gpus = list(nvsmi.get_gpus())
    gpulist = [str(gpu).split(' | ') for gpu in gpus]
    mem_util = [g[3] for g in gpulist]
    mem_util = [float(mem.split(':')[1].strip()[:-1]) for mem in mem_util]
    return np.argmin(mem_util)


def getMemUtil(msg='', gpu=0):
  print(msg, str(list(nvsmi.get_gpus())[gpu]).split(' | ')[3])
