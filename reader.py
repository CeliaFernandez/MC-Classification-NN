import time
import numpy as np
from matplotlib import pyplot as plt
import sys
import ROOT as ROOT
import math
from array import array
from utils import *

# -> Define the path
dirName = "Samples/"
#Sample1 = ["nanoLatino_TTTo2L2Nu__part1.root", "nanoLatino_TTTo2L2Nu__part2.root"]
#Sample2 = ["nanoLatino_DYJetsToLL_M-50-LO-ext1__part1.root"]


if len(sys.argv) < 2:
    sys.exit("Usage: You must specify at least a root file and the class number")

# -> Sample list
samples = []


for i in range(1, len(sys.argv) - 1):

    samples.append(sys.argv[i])


print(samples)
    
# -> Class number
class_n = sys.argv[-1]

#-> Get the files
files = [ROOT.TFile(dirName+fname) for fname in samples]


#-> Get the trees of each sample
Events = [f.Get("Events") for f in files]
 
# -> Number of entries
nn = [t.GetEntries() for t in Events]
NN = sum(nn) # Total entries in sample 1



# -> Training info
variableNames = ["nL", "ptl1", "ptl2", "dRll","lflav" ,"mll", "nj", "ptj1", "ptj2", "btag1", "btag2","MET_pt"]



print("----> Starting the construction of the samples\n")

print("The total number of files readed is :" + str(len(samples)))
print("The initial number of events per file is: \n")
for c,t in enumerate(Events):

    print("File "+str(c)+": "+str(t.GetEntries()))


# -> Output list of events
ev_list = []

print("Filtering Sample 1...")
ev_list = filterAndSave(class_n, Events)

   
# -> Redefinition of total events in sample

print("After the cuts the samples have been reduced to: \n")
print(str(NN)+"->" +str(len(ev_list)))


# -> Write the train and the test files

print("-> Writing samples in the txt files \n")

if str(class_n)[0:2]=='99':
    n_train = 1
    print("Test sample!!!! Only test file generated!")
else:
    n_train = 200000 # Number of samples in the train file (default)
    print("The number of events in the train file is (default): "+str(n_train))

writeTrainAndTest("train"+str(class_n)+".txt", "test"+str(class_n)+".txt", ev_list, n_train, class_n)



