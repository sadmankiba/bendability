from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chromosome import Chromosome
from util.reader import GeneReader
from util.util import IOUtil, PlotUtil, PathUtil
from .nucleosome import Nucleosome


class Genes:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._tr_df = GeneReader().read_transcription_regions_of(chrm.number)
        self._add_dyads_in_tr()

    def _add_dyads_in_tr(self) -> None:
        nucs = Nucleosome(self._chrm)
        # TODO: Nucs need not know about strand
        self._tr_df["dyads"] = self._tr_df.apply(
            lambda tr: nucs.dyads_between(tr["start"], tr["end"], tr["strand"]), axis=1
        )

    def _frwrd_tr_df(self) -> pd.DataFrame:
        return self._tr_df.query("strand == 1")

    def _rvrs_tr_df(self) -> pd.DataFrame:
        return self._tr_df.query("strand == -1")

    def plot_mean_c0_vs_dist_from_dyad(self) -> Path:
        frwrd_p1_dyads = self._frwrd_tr_df()["dyads"].apply(lambda dyads: dyads[0])
        frwrd_mean_c0 = self._chrm.mean_c0_around_bps(frwrd_p1_dyads, 600, 400)

        rvrs_p1_dyads = self._rvrs_tr_df()["dyads"].apply(lambda dyads: dyads[0])
        rvrs_mean_c0 = self._chrm.mean_c0_around_bps(rvrs_p1_dyads, 400, 600)[::-1]

        mean_c0 = (
            frwrd_mean_c0 * len(frwrd_p1_dyads) + rvrs_mean_c0 * len(rvrs_p1_dyads)
        ) / len(self._tr_df)

        plt.close()
        plt.clf()
        PlotUtil().show_grid()
        plt.plot(np.arange(-600, 400 + 1), mean_c0)

        plt.xlabel("Distance from dyad (bp)")
        plt.ylabel("Mean C0")
        plt.title(
            f"{self._chrm.c0_type} Mean C0 around +1 dyad"
            f" in chromosome {self._chrm.number}"
        )

        return IOUtil().save_figure(
            f"{PathUtil.get_figure_dir()}/gene/dist_p1_dyad_{self._chrm}.png"
        )

    def in_promoter(self, bps: np.ndarray | list[int] | pd.Series) -> np.ndarray:
        """
        Find whether some bps lies in promoter

        Promoter is defined as +-400bp from TSS
        """
        frwrd_prmtr_rgn = self._frwrd_tr_df()["start"].apply(
            lambda bp: np.arange(bp - 400, bp + 400 + 1)
        )
        rvrs_prmtr_rgn = self._rvrs_tr_df()["end"].apply(
            lambda bp: np.arange(bp - 400, bp + 400 + 1)
        )

        prmtr_rgn = np.concatenate(
            (
                np.array(frwrd_prmtr_rgn.tolist()).flatten(),
                np.array(rvrs_prmtr_rgn.tolist()).flatten(),
            )
        )

        return np.array([bp in prmtr_rgn for bp in bps])
