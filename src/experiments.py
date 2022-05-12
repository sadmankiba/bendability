from ast import Mod
from conformation.domains import BoundariesF, BoundariesFN, DomainsF, DomainsFN, BndFParm
from chromosome.chromosome import C0Spread, Chromosome
from chromosome.nucleosomes import Nucleosomes, Linkers
from feature_model.data_organizer import DataOrganizeOptions, SequenceLibrary, TrainTestSequenceLibraries
from feature_model.feat_selector import AllFeatureSelector, FeatureSelector
from models.prediction import Prediction
from motif.motifs import KMerMotifs, MotifsM35
from util.constants import RL, RL_LEN, TL, TL_LEN, GDataSubDir

from feature_model.model import ModelCat, ModelRunner

pred = Prediction(35)
chrm = Chromosome("VL", prediction=pred, spread_str=C0Spread.mcvr)
bndsf = BoundariesF(chrm, **BndFParm.SHR_50)
dmnsf = DomainsF(bndsf)
bnds = BoundariesFN(chrm, **BndFParm.SHR_50_LNK_0)
dmns = DomainsFN(bnds)
nucs = Nucleosomes(chrm)
lnks = Linkers(chrm)

bnds_l50 = bnds.extended(-50)
dmns_l50 = DomainsFN(bnds_l50)

class Experiments:
    @classmethod
    def bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(bndsf, dmnsf, GDataSubDir.BOUNDARIES)
    
    @classmethod 
    def nuc_lnk_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(lnks, nucs, GDataSubDir.NUCLEOSOMES)
    
    @classmethod
    def bnd_dmn_kmer_score(self):
        KMerMotifs.score(bnds_l50, dmns_l50, GDataSubDir.BOUNDARIES)
    
    @classmethod
    def lnk_nuc_kmer_score(self):
        KMerMotifs.score(lnks, nucs, GDataSubDir.BOUNDARIES)

    @classmethod 
    def ml_nn(self):
        libs = TrainTestSequenceLibraries(train=[SequenceLibrary(TL, TL_LEN)], test=[SequenceLibrary(RL, RL_LEN)])
        o = DataOrganizeOptions(k_list=[2,3,4])
        ModelRunner.run_model(libs, o, featsel=AllFeatureSelector(), cat=ModelCat.REGRESSOR)
    