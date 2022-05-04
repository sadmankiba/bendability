from conformation.domains import BoundariesFN, DomainsFN, BndFParm
from chromosome.chromosome import C0Spread, Chromosome
from models.prediction import Prediction
from motif.motifs import MotifsM35
from util.constants import GDataSubDir

class Experiments:
    @classmethod
    def bnd_dmn_V_z_test(self):
        pred = Prediction(35)
        chrm = Chromosome("VL", prediction=pred, spread_str=C0Spread.mcvr)
        bnds = BoundariesFN(chrm, **BndFParm.SHR_50_LNK_0)
        dmns = DomainsFN(bnds)
        m = MotifsM35()
        m.enrichment_compare(bnds, dmns, GDataSubDir.BOUNDARIES)