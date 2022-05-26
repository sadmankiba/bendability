import pandas as pd
from chromosome.regions import LEN
from conformation.domains import (
    BndParm,
    BndSel,
    BoundariesF,
    BoundariesFN,
    BoundariesType,
    DomainsF,
    DomainsFN,
    BndFParm,
)
from chromosome.chromosome import C0Spread, Chromosome
from chromosome.nucleosomes import Nucleosomes, Linkers
from feature_model.data_organizer import (
    DataOrganizeOptions,
    SequenceLibrary,
    TrainTestSequenceLibraries,
)
from chromosome.crossregions import SubRegions, sr_vl
from feature_model.feat_selector import AllFeatureSelector, FeatureSelector
from models.prediction import Prediction
from motif.motifs import KMerMotifs, MotifsM35
from util.util import FileSave, PathObtain
from util.constants import RL, RL_LEN, TL, TL_LEN, GDataSubDir, YeastChrNumList
from feature_model.model import ModelCat, ModelRunner


class Objs:
    # bndsfn_l50 = bndsfn.extended(-50)
    # dmnsfn_l50 = DomainsFN(bndsfn_l50)
    pass


class Experiments:
    @classmethod
    def chrm_stat(cls):
        t = []
        pred = Prediction(35)
        for c in YeastChrNumList:
            print(c)
            chrm = Chromosome(c, pred, C0Spread.mcvr)
            sr = SubRegions(chrm)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            nl = sr.bndrs.nearest_rgns(sr.lnkrs)
            t.append(nl[LEN].median())
        
        FileSave.tsv_gdatadir(pd.DataFrame({"t": t}), f"{GDataSubDir.TEST}/chrm.tsv")
        
    @classmethod
    def fasta(cls):
        sr_vl.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_WD_50)
        rg, d = sr_vl.dmns, GDataSubDir.DOMAINS
        FileSave.fasta(
            rg.seq(),
            f"{PathObtain.gen_data_dir()}/{d}/{sr_vl.chrm}_{rg}/seq.fasta",
        )

    @classmethod
    def lnk_bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(
            sr_vl.lnks_in_bndsf(), sr_vl.lnks_in_dmnsf(), GDataSubDir.LINKERS
        )

    @classmethod
    def bnd_dmn_V_z_test_all(self):
        pred = Prediction(35)
        for c in YeastChrNumList:
            chrm = Chromosome(c, pred, C0Spread.mcvr)
            sr = SubRegions(chrm)
            m = MotifsM35(c)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_WD_100)
            m.enrichment_compare(sr.bndrs, sr.dmns, GDataSubDir.BOUNDARIES)

    @classmethod
    def nuc_lnk_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(sr_vl.lnkrs, sr_vl.nucs, GDataSubDir.NUCLEOSOMES)

    @classmethod
    def kmer_score_lnks_bnd_dmn_V(self):
        KMerMotifs.score(
            sr_vl.lnks_in_bndsf(), sr_vl.lnks_in_dmnsf(), GDataSubDir.LINKERS
        )

    @classmethod
    def kmer_score_nucs_bnd_dmn_V(self):
        KMerMotifs.score(
            sr_vl.nucs_in_bndsf(), sr_vl.nucs_in_dmnsf(), GDataSubDir.NUCLEOSOMES
        )

    @classmethod
    def kmer_score_bnd_dmn(self):
        KMerMotifs.score(sr_vl.bndsf, sr_vl.dmnsf, GDataSubDir.BOUNDARIES)

    @classmethod
    def kmer_score_lnk_nuc(self):
        KMerMotifs.score(sr_vl.lnkrs, sr_vl.nucs, GDataSubDir.LINKERS)

    @classmethod
    def ml_nn(self):
        libs = TrainTestSequenceLibraries(
            train=[SequenceLibrary(TL, TL_LEN)], test=[SequenceLibrary(RL, RL_LEN)]
        )
        o = DataOrganizeOptions(k_list=[2, 3, 4])
        ModelRunner.run_model(
            libs, o, featsel=AllFeatureSelector(), cat=ModelCat.REGRESSOR
        )
