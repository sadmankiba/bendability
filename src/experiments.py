from conformation.domains import (
    BoundariesF,
    BoundariesFN,
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
from chromosome.crossregions import sr_vl
from feature_model.feat_selector import AllFeatureSelector, FeatureSelector
from models.prediction import Prediction
from motif.motifs import KMerMotifs, MotifsM35
from util.util import FileSave, PathObtain
from util.constants import RL, RL_LEN, TL, TL_LEN, GDataSubDir
from feature_model.model import ModelCat, ModelRunner


class Objs:
    bndsf = sr_vl.bndsf
    dmnsf = sr_vl.dmnsf
    bndsfn = sr_vl.bndsfn
    dmnsfn = sr_vl.dmnsfn
    nucs = sr_vl.nucs
    lnks = sr_vl.lnkrs
    # bndsfn_l50 = bndsfn.extended(-50)
    # dmnsfn_l50 = DomainsFN(bndsfn_l50)
    lnks_in_bndsf = sr_vl.lnks_in_bndsf
    lnks_in_dmnsf = sr_vl.lnks_in_dmnsf


class Experiments:
    @classmethod
    def fasta(cls):
        FileSave.fasta(
            Objs.dmnsf.seq(),
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.DOMAINS}/{Objs.chrm}_{Objs.dmnsf}/seq.fasta",
        )

    @classmethod
    def lnk_bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(
            Objs.lnks_in_bndsf,
            Objs.lnks_in_dmnsf,
            GDataSubDir.LINKERS
        )

    @classmethod
    def bnd_dmn_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(Objs.bndsf, Objs.dmnsf, GDataSubDir.BOUNDARIES)

    @classmethod
    def nuc_lnk_V_z_test(self):
        m = MotifsM35()
        m.enrichment_compare(Objs.lnks, Objs.nucs, GDataSubDir.NUCLEOSOMES)

    @classmethod 
    def kmer_score_lnk_bnd_dmn_V(self):
        KMerMotifs.score(Objs.lnks_in_bndsf, Objs.lnks_in_dmnsf, GDataSubDir.LINKERS)

    @classmethod
    def kmer_score_bnd_dmn(self):
        KMerMotifs.score(Objs.bndsf, Objs.dmnsf, GDataSubDir.BOUNDARIES)

    @classmethod
    def kmer_score_lnk_nuc(self):
        KMerMotifs.score(Objs.lnks, Objs.nucs, GDataSubDir.LINKERS)

    @classmethod
    def ml_nn(self):
        libs = TrainTestSequenceLibraries(
            train=[SequenceLibrary(TL, TL_LEN)], test=[SequenceLibrary(RL, RL_LEN)]
        )
        o = DataOrganizeOptions(k_list=[2, 3, 4])
        ModelRunner.run_model(
            libs, o, featsel=AllFeatureSelector(), cat=ModelCat.REGRESSOR
        )
