""" Methods used to create the samples """
import ROOT
import math
from array import array



def phiDifference(phi1, phi2):

    deltaPhi = abs(phi1 - phi2)
    if (deltaPhi > 3.14):
        deltaPhi = 2*3.14 - deltaPhi

    return deltaPhi


def angularDistance(deltaEta, deltaPhi):

    return math.sqrt(deltaEta**2 + deltaPhi**2)



def accessEntry(events, n):

        # n: number of the entry

        """ -> Variable definition:
                nL: Number of leptons
                ptl1: pt of the leading lepton
                ptl2: pt of the subleading lepton 
                dRll: Angular distance between lepton
                mll: Dilepton invariant mass
                nj: Number of jets
                dRlf: Angular distance between leading lepton and leading jet   
                ptj1: pt of the leaing jet
                ptj2: pt of the subleading jet 
                dRjj: Angular distance between leading jet and subleading jet
                MET_pt: MET pt
                MET_phi: MET phi
        """

        events.GetEntry(n)

        entry = {} # Variable entry
        variableNames = ["nL", "ptl1", "ptl2", "dRll","lflav" ,"mll", "nj", "ptj1", "ptj2", "btag1", "btag1","MET_pt"]

        # Lepton variables:
        entry["nL"] = events.nLepton
        entry["ptl1"] = events.Lepton_pt[0]
        if (entry["nL"] > 1):
            entry["ptl2"] = events.Lepton_pt[1]
            dEta = abs(events.Lepton_eta[0]-events.Lepton_eta[1])
            dPhi = phiDifference(events.Lepton_phi[0], events.Lepton_phi[1])
            entry["dRll"] = angularDistance(dEta, dPhi)
            entry["mll"] = events.mll

        else:
            entry["ptl2"] = -10
            entry["dRll"] = -10
            entry["mll"] = -10

        if (events.Lepton_electronIdx[0]==-1 and events.Lepton_electronIdx[1] ==-1):
            entry["lflav"] = 1 # muon flavor
        elif (events.Lepton_muonIdx[0]==-1 and events.Lepton_muonIdx[1] ==-1):
            entry["lflav"] = 1 # electron flavor
        else:
            entry["lflav"] = 2 # electron+ muon flavor

        # Jet variables:
        entry["nj"] = events.nJet
        if(entry["nj"] > 0):
            entry["ptj1"] = events.Jet_pt[0]
            entry["btag1"] = events.Jet_btagCSVV2[0] if events.Jet_btagCSVV2[0] != -10 else 0
            
        else:
            entry["ptj1"] = -10
            entry["btag1"] = -10
            
        if (entry["nj"] > 1):
            entry["ptj2"] = events.Jet_pt[1]
            entry["btag2"] = events.Jet_btagCSVV2[1] if events.Jet_btagCSVV2[1] != -10 else 0                               

        else:
            entry["ptj2"] = -10
            
        # MET variables
        entry["MET_pt"] = events.MET_pt

        return entry


def filterAndSave(sample, events):

    ev_list = []
    variableNames = ["nL", "ptl1", "ptl2", "dRll","lflav" ,"mll", "nj", "ptj1", "ptj2", "btag1", "btag1","MET_pt"]

    # Define a tree with the filtered events and variables
    f = ROOT.TFile("Sample"+str(sample)+"filtered.root", 'recreate')
    output_tree = ROOT.TTree("Events", "Events")
    # Variables
    nL = array('i',[0])
    ptl1, ptl2 = array('d',[0.]), array('d',[0.])
    dRll = array('d',[0.])
    mll = array('d',[0.])
    lflav = array('i',[0])
    nj = array('i',[0])
    ptj1, ptj2 = array('d',[0.]), array('d',[0.])
    btag1, btag2 = array('d',[0.]), array('d',[0.])
    MET_pt = array('d',[0.])
    # Branches
    output_tree.Branch('nL', nL, 'nL/I')
    output_tree.Branch('ptl1', ptl1, 'ptl1/D')
    output_tree.Branch('ptl2', ptl2, 'ptl2/D')
    output_tree.Branch('dRll', dRll, 'dRll/D')
    output_tree.Branch('mll', mll, 'mll/D')
    output_tree.Branch('lflav', lflav, 'lflav/I')
    output_tree.Branch('nj', nj, 'nj/I')
    output_tree.Branch('ptj1', ptj1, 'ptj1/D')
    output_tree.Branch('ptj2', ptj2, 'ptj2/D')
    output_tree.Branch('btag1', btag1, 'btag1/D')
    output_tree.Branch('btag2', btag2, 'btag2/D')
    output_tree.Branch('MET_pt', MET_pt, 'MET_pt/D')


    for c,t in enumerate(events):

        #for i in range(0, t.GetEntries()):    
        for i in range(0, t.GetEntries()):

            t.GetEntry(i) # access the tree

            progress = printProgress(i,t.GetEntries())

            if (progress!= -1): print("Sample "+str(sample)+", Tree "+str(c+1)+"/"+str(len(events))+": "+ progress+ " filtered")
    
            if (t.nLepton < 2): continue
            if (t.nJet < 2): continue
            
            entry = accessEntry(t, i)
            ev_list.append(entry)   
 
            nL[0] = entry['nL']
            ptl1[0] = entry['ptl1']
            ptl2[0] = entry['ptl2']
            dRll[0] = entry['dRll']
            mll[0] = entry['mll']
            lflav[0] = entry['lflav']
            nj[0] = entry['nj']
            ptj1[0] = entry['ptj1']
            ptj2[0] = entry['ptj2']
            btag1[0] = entry['btag1']
            btag2[0] = entry['btag2']
            MET_pt[0] = entry['MET_pt']

            #print(nL, ptl1, entry['nL'], entry['ptl1'])

            output_tree.Fill()

            
    f.Write()
    f.Close()    
    
    return ev_list



def returnEntryLine(entry, VariableNames):

    # Return the line which is going to be writing in the file        

    line = "" # empty line

    for n in range(0, len(VariableNames)):

        name = VariableNames[n]
        line = line + str(entry[name])
        if (n != len(VariableNames)-1):
            line = line + ","

    return line


def writeTrainAndTest(train_name, test_name, entry_list, train_number, sample):

    train = open(train_name, "w").close() # erase previous train file
    test = open(test_name, "w").close() # erase previous test file

    max_n = len(entry_list)
    
    variableNames = ["nL", "ptl1", "ptl2", "dRll","lflav" ,"mll", "nj", "ptj1", "ptj2", "btag1", "btag1","MET_pt"]

    for n in range(0, max_n):

        progress = printProgress(n,max_n)

        if (progress!= -1): print("Sample "+str(sample)+" files written: "+ progress)

        entry = entry_list[n]
        line = returnEntryLine(entry, variableNames)

        file_name = train_name if (n < train_number) else test_name

        outputfile = open(file_name, "a")
        outputfile.write(line)
        outputfile.write("\n")
        outputfile.close()


def printProgress(n, n_max):
    
    percentajes = [float(i) for i in range(0, 101, 5)]
    marks = [int(n_max*i/100) for i in range(0, 101, 5)]
   
    
    if n in marks:

        progress = str(percentajes[marks.index(n)])+"%"
        return progress

    else:
        return -1

