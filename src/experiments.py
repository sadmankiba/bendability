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

class Objs:
    pred = Prediction(35)
    chrm = Chromosome("VL", prediction=pred, spread_str=C0Spread.mcvr)
    bndsf = BoundariesF(chrm, **BndFParm.SHR_50)
    dmnsf = DomainsF(bndsf)
    bndsfn = BoundariesFN(chrm, **BndFParm.SHR_50_LNK_0)
    dmns = DomainsFN(bndsfn)
    nucs = Nucleosomes(chrm)
    lnks = Linkers(chrm)
    bndsfn_l50 = bndsfn.extended(-50)
    dmnsfn_l50 = DomainsFN(bndsfn_l50)
    

class Experiments:
    @classmethod
    def bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(Objs.bndsf, Objs.dmnsf, GDataSubDir.BOUNDARIES)
    
    @classmethod 
    def nuc_lnk_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(Objs.lnks, Objs.nucs, GDataSubDir.NUCLEOSOMES)
    
    @classmethod
    def bnd_dmn_kmer_score(self):
        KMerMotifs.score(Objs.bndsfn_l50, Objs.dmnsfn_l50, GDataSubDir.BOUNDARIES)
    
    @classmethod
    def lnk_nuc_kmer_score(self):
        KMerMotifs.score(Objs.lnks, Objs.nucs, GDataSubDir.BOUNDARIES)

    @classmethod 
    def ml_nn(self):
        libs = TrainTestSequenceLibraries(train=[SequenceLibrary(TL, TL_LEN)], test=[SequenceLibrary(RL, RL_LEN)])
        o = DataOrganizeOptions(k_list=[2,3,4])
        ModelRunner.run_model(libs, o, featsel=AllFeatureSelector(), cat=ModelCat.REGRESSOR)
    