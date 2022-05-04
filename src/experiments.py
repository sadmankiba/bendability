from conformation.domains import BoundariesFN, DomainsFN, BndFParm
from chromosome.chromosome import C0Spread, Chromosome
from chromosome.nucleosomes import Nucleosomes, Linkers
from models.prediction import Prediction
from motif.motifs import KMerMotifs, MotifsM35
from util.constants import GDataSubDir

pred = Prediction(35)
chrm = Chromosome("VL", prediction=pred, spread_str=C0Spread.mcvr)
bnds = BoundariesFN(chrm, **BndFParm.SHR_50_LNK_0)
dmns = DomainsFN(bnds)
nucs = Nucleosomes(chrm)
lnks = Linkers(chrm)

class Experiments:
    @classmethod
    def bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(bnds, dmns, GDataSubDir.BOUNDARIES)
    
    @classmethod 
    def nuc_lnk_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(lnks, nucs, GDataSubDir.NUCLEOSOMES)
    
    @classmethod
    def bnd_dmn_kmer_score(self):
        KMerMotifs.score(bnds, dmns, GDataSubDir.BOUNDARIES)