# Script to process MiniAODs and convert them into flat ntuples

# Import section

import ROOT # for 4-vector builds
from DataFormats.FWLite import Events, Handle # to open MiniAODs

import os

# parser for arguments

import argparse

# Mapping for pdgID

pdgId_map = {
    553: 'upsilon_id',
    15: 'tau_plus' ,
    -15: 'tau_minus',
    211: 'pion_plus' ,
    -211: 'pion_minus',
    16: 'tau_neu_plus' ,
    -16: 'tau_neu_minus',
    111: 'neutral_pion_id' ,
    22: 'photon_id' ,
}
# Function definitions 


# Main 

if __name__ == '__main__':


    # Argument parser and fixing the CMSSW version via the options container
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--f_in', default='UpsilonToTauTau_PUPoissonAve20_102X_upgrade2018_realistic_v18_3prong_m15_miniaod_part0') 
    #parser.add_argument('--maxEvents', default = 100)
    args = parser.parse_args()

    # file path
    path = '/isilon/data/users/mstamenk/tau-project-summer24/samples/preprocess15GeVMiniaodSampleWITHOUTpTCutIncl'
    filename = path + '/' + args.f_in + '.root'

    from FWCore.ParameterSet.VarParsing import VarParsing # Needed to input the file
    options = VarParsing ('python')  
    options.inputFiles = [filename]
    options.maxEvents =  2 # run on 10 events first, -1 for all the events

    options.parseArguments()
    #print(options)

    # Labels and handles

    handleGen  = Handle ("std::vector<reco::GenParticle>") # CMSSW list of reco::GenParticles
    labelGen = ("prunedGenParticles")

    handleReco = Handle ("std::vector<pat::PackedCandidate>") # CMSSW list of reconstructed candidates
    recoLabel = ("packedPFCandidates")

    lostLabel = ("lostTracks")

    handleMET = Handle ("std::vector<pat::MET>")
    labelMET = ("slimmedMETs")


    # Open the events in MiniAOD

    events = Events(options)

    eventNumber = 0

    for event in events: # Loops over all the events sepcified with maxEvents
        print("Event number",eventNumber)
        event.getByLabel(labelGen, handleGen)
        gen_particles = handleGen.product()

        event.getByLabel(recoLabel, handleReco)
        pf_particles = handleReco.product()

        event.getByLabel(lostLabel, handleReco)
        lost_particles = handleReco.product()
    
        event.getByLabel(labelMET, handleMET)
        met = handleMET.product().front()

        gen_dict = {}

        print(len(gen_particles))
        for gen_particle in gen_particles:
            if gen_particle.pdgId() in pdgId_map.keys():
                print(pdgId_map[int(gen_particle.pdgId())])
                print(gen_particle.pt(), gen_particle.eta(), gen_particle.phi(), gen_particle.mass())

                vec = ROOT.TLorentzVector()
                vec.SetPtEtaPhiM(gen_particle.pt(), gen_particle.eta(), gen_particle.phi(), gen_particle.mass())
                gen_dict[pdgId_map[int(gen_particle.pdgId())]] = vec

            #else: 
                #print(gen_particle.pdgId())

        # Retrive 4 vectors
        tau_plus = gen_dict['tau_plus']  
        tau_minus = gen_dict['tau_minus']
        upsilon = gen_dict['upsilon_id']

        print("Masses tau+, tau-, upsilon sum, upsilon id", tau_plus.M(), tau_minus.M(), (tau_plus+tau_minus).M(), upsilon.M())

        print("Print deltaR", tau_plus.DeltaR(tau_minus))



        #print(len(pf_particles))
        #print(len(lost_particles))
        #print(met)





