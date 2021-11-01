from __future__ import annotations
from pathlib import Path
from typing import Literal, NamedTuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chromosome.chromosome import Chromosome, MultiChrm
from chromosome.genes import Genes, Promoters
from models.prediction import Prediction
from util.util import FileSave, PlotUtil, PathObtain, Attr
from util.custom_types import ChrId, PositiveInt, NonNegativeInt, PosOneIdx
from util.constants import ChrIdList


# TODO: Inherit dataframe wrapper

LEFT = "left"
RIGHT = "right"
MIDDLE = "middle"
SCORE = "score"
IN_PROMOTER = "in_promoter"
MEAN_C0 = "mean_c0"


class BoundaryNT(NamedTuple):
    left: PosOneIdx
    right: PosOneIdx
    middle: PosOneIdx
    score: float


class BoundariesHE:
    """
    Represenation of boundaries in a chromosome found with Hi-C Explorer

    Note: Here domains are also determined by boundaries
    """

    def __init__(
        self, chrm: Chromosome, res: PositiveInt = 500, lim: PositiveInt = 250
    ):
        """
        Args:
            lim: Limit around boundary middle bp to include in boundary
        """
        # TODO: Make bndrs_df private
        self._chrm = chrm
        self.res = res
        self.bndrs_df = self._read_boundaries()
        self.lim = lim
        self._mean_c0 = None 
        self._prmtr_bndrs = None 
        self._np_bndrs = None
        self._add_mean_c0_col()
        self._add_in_promoter_col()

    def __iter__(self) -> Iterable[BoundaryNT]:
        return self.bndrs_df.itertuples()

    def __getitem__(self, key: NonNegativeInt | str) -> pd.Series:
        if isinstance(key, NonNegativeInt):
            return self.bndrs_df.iloc[key]

        if key in self.bndrs_df.columns:
            return self.bndrs_df[key]

        raise KeyError

    def __len__(self) -> int:
        return len(self.bndrs_df)
        
    def __str__(self):
        return f"res_{self.res}_lim_{self.lim}"

    def _read_boundaries(
        self,
    ) -> pd.DataFrame[LEFT:int, RIGHT:int, MIDDLE:int, SCORE:float]:
        df = pd.read_table(
            f"{PathObtain.input_dir()}/domains/"
            f"{self._chrm.number}_res_{self.res}_hicexpl_boundaries.bed",
            delim_whitespace=True,
            header=None,
            names=["chromosome", LEFT, RIGHT, "id", SCORE, "_"],
        )
        return df.assign(
            middle=lambda df: ((df[LEFT] + df[RIGHT]) / 2).astype(int)
        ).drop(columns=["chromosome", "id", "_"])

    def _add_mean_c0_col(self) -> None:
        """Add mean c0 of each bndry"""
        self.bndrs_df = self.bndrs_df.assign(
            mean_c0=lambda df: self._chrm.mean_c0_at_bps(
                df[MIDDLE], self.lim, self.lim
            )
        )

    def _add_in_promoter_col(self) -> None:
        self.bndrs_df = self.bndrs_df.assign(
            in_promoter=lambda df: Promoters(self._chrm).is_in_prmtr(df[MIDDLE])
        )

    def cover_mask(self) -> np.ndarray:
        return self._chrm.get_cvr_mask(self.bndrs_df[MIDDLE], self.lim, self.lim)

    @property
    def mean_c0(self) -> float:
        def calc_mean_c0():
            return self._chrm.c0_spread()[self.cover_mask()].mean()

        return Attr.calc_attr(self, "_mean_c0", calc_mean_c0)

    def prmtr_bndrs(self) -> BoundariesHE:
        def calc_prmtr_bndrs():
            bndrs = BoundariesHE(self._chrm, self.res, self.lim)
            bndrs.bndrs_df = pd.DataFrame(
                [bndry for bndry in self if getattr(bndry, IN_PROMOTER)]
            )
            return bndrs

        return Attr.calc_attr(self, "_prmtr_bndrs", calc_prmtr_bndrs)

    def non_prmtr_bndrs(self) -> BoundariesHE:
        def calc_np_bndrs():
            bndrs = BoundariesHE(self._chrm, self.res, self.lim)
            bndrs.bndrs_df = pd.DataFrame(
                [bndry for bndry in self if not getattr(bndry, IN_PROMOTER)]
            )
            return bndrs

        return Attr.calc_attr(self, "_np_bndrs", calc_np_bndrs)


START = "start"
END = "end"
LEN = "len"
MEAN_C0 = "mean_c0"


class DomainNT:
    start: PosOneIdx
    end: PosOneIdx
    len: int
    mean_c0: float


class DomainsHE:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._boundaries = BoundariesHE(chrm)
        self._domains = self._get_domains()
        self._add_mean_c0()

    def __iter__(self) -> Iterable[NamedTuple[START:int, END:int, LEN:int]]:
        return self._domains.itertuples()

    def __getitem__(self, key: str) -> pd.Series:
        if key in self._domains.columns:
            return self._domains[key]

        raise KeyError

    def _add_mean_c0(self) -> None:
        self._domains[MEAN_C0] = self._domains.apply(
            lambda domain: self._chrm.mean_c0_segment(domain[START], domain[END]),
            axis=1,
        )

    @property
    def mean_c0(self) -> float:
        if not hasattr(self, "_mean_c0"):
            c0_spread = self._chrm.c0_spread()
            self._mean_c0 = c0_spread[~self._boundaries.cover_mask()].mean()

        return self._mean_c0

    def _get_domains(self) -> pd.DataFrame[START:int, END:int, LEN:int]:
        return pd.DataFrame(
            {
                "start": self._boundaries[RIGHT].tolist()[:-1],
                "end": self._boundaries[LEFT].tolist()[1:],
            }
        ).assign(len=lambda df: df["end"] - df["start"])



class PlotBoundariesHE:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self._bndrs = BoundariesHE(chrm)
        self._figsubdir = "domains"

    def density(self):
        PlotUtil.show_grid()
        sns.distplot(self._bndrs[MEAN_C0], hist=False, kde=True, label="boundaries")
        prmtrs = Promoters(self._chrm)
        sns.distplot(prmtrs[MEAN_C0], hist=False, kde=True, label="promoters")
        sns.distplot(self._bndrs.prmtr_bndrs()[MEAN_C0], hist=False, kde=True, label="prm boundaries")
        sns.distplot(self._bndrs.non_prmtr_bndrs()[MEAN_C0], hist=False, kde=True, label="nonprm boundaries")
        prmtrs_with_bndrs = prmtrs.and_x(self._bndrs[MIDDLE], True)
        prmtrs_wo_bndrs = prmtrs.and_x(self._bndrs[MIDDLE], False)
        sns.distplot(prmtrs_with_bndrs[MEAN_C0], hist=False, kde=True, label="promoters with bndry")
        sns.distplot(prmtrs_wo_bndrs[MEAN_C0], hist=False, kde=True, label="promoters w/o bndry")
        plt.legend()
        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/boundaries_density_c0_{self._bndrs}_{prmtrs}_{self._chrm}.png"
        )

    def line_c0_around(self, pltlim=250):
        bndrs_mid = self._bndrs[MIDDLE]
        pbndrs_mid = self._bndrs.prmtr_bndrs()[MIDDLE]
        npbndrs_mid = self._bndrs.non_prmtr_bndrs()[MIDDLE]

        mc0_bndrs = self._chrm.mean_c0_around_bps(bndrs_mid, pltlim, pltlim)
        mc0_pbndrs = self._chrm.mean_c0_around_bps(pbndrs_mid, pltlim, pltlim)
        mc0_npbndrs = self._chrm.mean_c0_around_bps(npbndrs_mid, pltlim, pltlim)

        PlotUtil.clearfig()

        x = np.arange(2 * pltlim + 1) - pltlim
        plt.plot(x, mc0_bndrs, color="tab:green", label="all")
        plt.plot(x, mc0_pbndrs, color="tab:orange", label="promoter")
        plt.plot(x, mc0_npbndrs, color="tab:blue", label="non-promoter")

        self._chrm.plot_avg()

        plt.legend()
        PlotUtil.show_grid()
        plt.xlabel("Distance from boundary middle(bp)")
        plt.ylabel("C0")
        plt.title(
            f"Mean {self._chrm.c0_type} C0 around boundaries in chromosome {self._chrm.number}"
        )

        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/mean_c0_bndrs_{self._chrm._chr_id}_plt_{pltlim}.png"
        )

    def line_c0_around_indiv(self) -> None:
        for bndry in self._bndrs.prmtr_bndrs():
            self._line_c0_around_indiv(bndry, "prmtr")

        for bndry in self._bndrs.non_prmtr_bndrs():
            self._line_c0_around_indiv(bndry, "nonprmtr")

    def _line_c0_around_indiv(self, bndry: BoundaryNT, pstr: str):
        mid = getattr(bndry, MIDDLE)
        lim = self._bndrs.lim
        self._chrm.plot_moving_avg(mid - lim, mid + lim)
        plt.xticks(ticks=[mid - lim, mid, mid + lim], labels=[-lim, 0, +lim])
        plt.xlabel(f"Distance from boundary middle")
        plt.ylabel("Intrinsic Cyclizability")
        plt.title(
            f"C0 around {pstr} boundary at {mid} bp of chromosome {self._chrm.number}."
            f"Found with res {self._bndrs.res}"
        )

        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/{self._chrm._chr_id}/{pstr}_indiv_{mid}.png"
        )

    def scatter_mean_c0_at_indiv(self) -> Path:
        markers = ["o", "s"]
        labels = ["promoter", "non-promoter"]
        colors = ["tab:blue", "tab:orange"]

        PlotUtil.clearfig()
        PlotUtil.show_grid()

        p_x = self._bndrs.prmtr_bndrs()[MIDDLE]
        np_x = self._bndrs.non_prmtr_bndrs()[MIDDLE]
        plt.scatter(
            p_x,
            self._bndrs.prmtr_bndrs()[MEAN_C0],
            marker=markers[0],
            label=labels[0],
            color=colors[0],
        )
        plt.scatter(
            np_x,
            self._bndrs.non_prmtr_bndrs()[MEAN_C0],
            marker=markers[1],
            label=labels[1],
            color=colors[1],
        )

        horiz_colors = ["tab:green", "tab:red", "tab:purple"]
        chrm_mean_c0 = self._chrm.c0_spread().mean()
        PlotUtil.horizline(DomainsHE(self._chrm).mean_c0, horiz_colors[0], "domains")
        PlotUtil.horizline(chrm_mean_c0, horiz_colors[1], "chromosome")
        PlotUtil.horizline(self._bndrs.mean_c0, horiz_colors[2], "boundaries")

        plt.xlabel("Position along chromosome (bp)")
        plt.ylabel("Mean C0")
        plt.title(
            f"Comparison of mean {self._chrm.c0_type} C0 among boundaries"
            f" in chromosome {self._chrm.number}"
        )
        plt.legend()

        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/mean_c0_scatter_{self._bndrs}.png"
        )


class BoundariesDomainsHEQuery:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._boundaries = BoundariesHE(chrm)
        self._domains = DomainsHE(chrm)

    def prmtr_bndrs_mean_c0(self) -> float:
        return self._chrm.mean_c0_of_segments(
            self._boundaries.prmtr_bndrs()[MIDDLE],
            self._boundaries._lim,
            self._boundaries._lim,
        )

    def non_prmtr_bndrs_mean_c0(self) -> float:
        return self._chrm.mean_c0_of_segments(
            self._boundaries.non_prmtr_bndrs()[MIDDLE],
            self._boundaries._lim,
            self._boundaries._lim,
        )

    def num_bndry_mean_c0_greater_than_dmn(self) -> PositiveInt:
        return (self._boundaries[MEAN_C0] > self._domains.mean_c0).sum()

    def num_prmtr_bndry_mean_c0_greater_than_dmn(self) -> float:
        return (self._boundaries.prmtr_bndrs()[MEAN_C0] > self._domains.mean_c0).sum()

    def num_non_prmtr_bndry_mean_c0_greater_than_dmns(self) -> float:
        return (
            self._boundaries.non_prmtr_bndrs()[MEAN_C0] > self._domains.mean_c0
        ).sum()


class MCBoundariesHE:
    def __init__(self, mchrm: MultiChrm):
        self._mchrm = mchrm
        self._bndrs = list(map(lambda c: BoundariesHE(c), mchrm))

    def __iter__(self):
        return iter(self._bndrs)

    def __str__(self):
        return str(self._mchrm)


class MCDomainsHE:
    def __init__(self, mchrm: MultiChrm):
        self._mchrm = mchrm
        self._dmns = list(map(lambda c: DomainsHE(c), mchrm))

    def __iter__(self):
        return iter(self._dmns)

    def __str__(self):
        return str(self._mchrm)


# TODO:
# Rewrite by using MCBoundaries and removing unnecessary procedures
class MCBoundariesHECollector:
    def __init__(
        self,
        prediction: Prediction,
        chrids: tuple[ChrId] = ChrIdList,
        res: PositiveInt = 500,
        lim: PositiveInt = 250,
    ):
        self._prediction = prediction
        self._chrids = chrids
        self._res = res
        self._lim = lim
        self._coll_df = pd.DataFrame({"ChrID": chrids})
        self._chrms = self._get_chrms()
        self._mc_bndrs = self._get_mc_bndrs()

    def __str__(self):
        ext = "with_vl" if "VL" in self._chrids else "without_vl"
        return f"res_{self._res}_lim_{self._lim}_md_{self._prediction}_{ext}"

    def _get_chrms(self) -> pd.Series:
        return pd.Series(
            list(
                map(lambda chrm_id: Chromosome(chrm_id, self._prediction), self._chrids)
            )
        )

    def _get_mc_bndrs(self) -> pd.Series[BoundariesHE]:
        return self._chrms.apply(lambda chrm: BoundariesHE(chrm, self._res, self._lim))

    def _add_num_bndrs(self) -> Literal["num_bndrs"]:
        num_bndrs_col = "num_bndrs"
        if num_bndrs_col not in self._coll_df.columns:
            self._coll_df[num_bndrs_col] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df)
            )

        return num_bndrs_col

    def _add_num_dmns(self) -> str:
        num_dmns_col = "num_dmns"
        if num_dmns_col not in self._coll_df.columns:
            num_bndrs_col = self._add_num_bndrs()
            self._coll_df[num_dmns_col] = self._coll_df[num_bndrs_col] - 1

        return num_dmns_col

    def _add_bndrs_mean(self) -> Literal["c0_bndrs"]:
        c0_bndrs_col = "c0_bndrs"
        if c0_bndrs_col not in self._coll_df.columns:
            self._coll_df["c0_bndrs"] = self._mc_bndrs.apply(
                lambda bndrs: bndrs.bndry_domain_mean_c0()[0]
            )

        return c0_bndrs_col

    def _add_dmns_mean(self) -> Literal["c0_dmns"]:
        c0_dmns_col = "c0_dmns"
        if c0_dmns_col not in self._coll_df.columns:
            self._coll_df[c0_dmns_col] = self._mc_bndrs.apply(
                lambda bndrs: bndrs.bndry_domain_mean_c0()[1]
            )

        return c0_dmns_col

    def _add_num_bndrs_gt_dmns(self) -> str:
        num_bndrs_gt_dmns_col = "num_bndrs_gt_dmns"
        if num_bndrs_gt_dmns_col not in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[num_bndrs_gt_dmns_col] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return num_bndrs_gt_dmns_col

    def _add_num_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 5
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df.query("in_promoter")["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return self.col_for(method_id)

    def _add_num_non_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 6
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df.query("not in_promoter")["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return self.col_for(method_id)

    def _add_num_prmtr_bndrs(self) -> Literal["num_p_b"]:
        method_id = 7
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query("in_promoter"))
            )

        return self.col_for(method_id)

    def _add_num_non_prmtr_bndrs(self) -> Literal["num_np_b"]:
        method_id = 8
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query("not in_promoter"))
            )

        return self.col_for(method_id)

    def col_for(self, method_id: int) -> str:
        col_map = {
            0: "num_bndrs",
            1: "num_dmns",
            2: "c0_bndrs",
            3: "c0_dmns",
            4: "num_bndrs_gt_dmns",
            5: "num_p_b_gt_d",
            6: "num_np_b_gt_d",
            7: "num_p_b",
            8: "num_np_b",
        }
        return col_map[method_id]

    def save_stat(self, methods: list[int] = None) -> Path:
        method_map = {
            0: self._add_num_bndrs,
            1: self._add_num_dmns,
            2: self._add_bndrs_mean,
            3: self._add_dmns_mean,
            4: self._add_num_bndrs_gt_dmns,
            5: self._add_num_prmtr_bndrs_gt_dmns,
            6: self._add_num_non_prmtr_bndrs_gt_dmns,
            7: self._add_num_prmtr_bndrs,
            8: self._add_num_non_prmtr_bndrs,
        }
        for m in methods:
            method_map[m]()

        self._coll_df["res"] = np.full((len(self._coll_df),), self._res)
        self._coll_df["lim"] = np.full((len(self._coll_df),), self._lim)
        self._coll_df["model"] = np.full((len(self._coll_df),), str(self._prediction))

        return FileSave.append_tsv(
            self._coll_df,
            f"{PathObtain.data_dir()}/generated_data/mcdomains/mcdmns_stat.tsv",
        )

    def plot_scatter_mean_c0(self) -> Path:
        """Draw scatter plot of mean c0 at boundaries and domains of
        chromosomes"""
        chrms = self._chrms
        chrm_means = chrms.apply(lambda chrm: chrm.c0_spread().mean())

        mc_bndrs = self._mc_bndrs
        mc_prmtr_bndrs_c0 = mc_bndrs.apply(lambda bndrs: bndrs.prmtr_bndrs_mean_c0())
        mc_non_prmtr_bndrs_c0 = mc_bndrs.apply(
            lambda bndrs: bndrs.non_prmtr_bndrs_mean_c0()
        )

        mc_bndrs_dmns_c0 = mc_bndrs.apply(lambda bndrs: bndrs.bndry_domain_mean_c0())
        mc_bndrs_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:, 0]
        mc_dmns_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:, 1]

        # Print comparison
        print("bndrs > dmns:", (mc_bndrs_c0 > mc_dmns_c0).sum())
        print("prmtr bndrs > dmns:", (mc_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print("non prmtr bndrs > dmns:", (mc_non_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print("chrms > dmns:", (chrm_means.to_numpy() > mc_dmns_c0).sum())

        PlotUtil.show_grid()
        x = np.arange(len(self._chrids))
        markers = ["o", "s", "p", "P", "*"]
        labels = [
            "chromosome",
            "promoter bndrs",
            "non-promoter bndrs",
            "boundaries",
            "domains",
        ]

        for i, y in enumerate(
            (
                chrm_means,
                mc_prmtr_bndrs_c0,
                mc_non_prmtr_bndrs_c0,
                mc_bndrs_c0,
                mc_dmns_c0,
            )
        ):
            plt.scatter(x, y, marker=markers[i], label=labels[i])

        plt.xticks(x, self._chrids)
        plt.xlabel("Chromosome")
        plt.ylabel("Mean C0")
        plt.title(f"Comparison of mean C0 in boundaries vs. domains")
        plt.legend()

        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcdomains/bndrs_dmns_c0_{self}.png"
        )

    def plot_bar_perc_in_prmtrs(self) -> Path:
        chrms = self._chrms
        mc_bndrs = chrms.apply(lambda chrm: BoundariesHE(chrm, self._res))
        perc_in_prmtrs = mc_bndrs.apply(
            lambda bndrs: Promoters(bndrs._chrm)
            .is_in_prmtr(bndrs.bndrs_df["middle"])
            .mean()
            * 100
        )

        PlotUtil.show_grid()
        x = np.arange(len(self._chrids))
        plt.bar(x, perc_in_prmtrs)
        plt.xticks(x, self._chrids)
        plt.xlabel("Chromosome")
        plt.ylabel("Boundaries in promoters (%)")
        plt.title(f"Percentage of boundaries in promoters in chromosomes")
        plt.legend()
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcdomains/perc_bndrs_in_promoters_{self}.png"
        )

    def num_bndrs_dmns(self) -> tuple[float, float]:
        mc_bndrs = self._mc_bndrs
        num_bndrs = mc_bndrs.apply(lambda bndrs: len(bndrs.bndrs_df)).sum()
        num_dmns = num_bndrs - len(self._chrids)
        return num_bndrs, num_dmns

    def mean_dmn_len(self) -> float:
        mc_bndrs = self._mc_bndrs
        return (
            mc_bndrs.apply(lambda bndrs: bndrs.get_domains()["len"].sum()).sum()
            / self.num_bndrs_dmns()[1]
        )

    def individual_bndry_stat(self) -> None:
        # TODO: Reduce function calls. Access index with .name if needed.
        mc_bndrs = self._mc_bndrs
        num_mc_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_bndry_mean_c0_greater_than_dmn(bndrs)
        ).sum()
        print("num_mc_bndrs_gt", num_mc_bndrs_gt)
        num_mc_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_prmtr_bndry_mean_c0_greater_than_dmn(bndrs)
        ).sum()
        print("num_mc_prmtr_bndrs_gt", num_mc_prmtr_bndrs_gt)

        num_mc_non_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_non_prmtr_bndry_mean_c0_greater_than_dmns(
                bndrs
            )
        ).sum()
        print("num_mc_non_prmtr_bndrs_gt", num_mc_non_prmtr_bndrs_gt)

        num_mc_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query("in_promoter"))
        ).sum()
        print("num_mc_prmtr_bndrs", num_mc_prmtr_bndrs)

        num_mc_non_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query("not in_promoter"))
        ).sum()
        print("num_mc_non_prmtr_bndrs", num_mc_non_prmtr_bndrs)


class MCBoundariesHEAggregator:
    def __init__(self, coll: MCBoundariesHECollector):
        self._coll = coll
        self._agg_df = pd.DataFrame({"ChrIDs": [coll._coll_df["ChrID"].tolist()]})

    def _bndrs_gt_dmns(self):
        self._coll.save_stat([0, 4])
        self._agg_df["b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(4)].sum()
            / self._coll._coll_df[self._coll.col_for(0)].sum()
            * 100
        )

    def _p_bndrs_gt_dmns(self):
        self._coll.save_stat([7, 5])
        self._agg_df["p_b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(5)].sum()
            / self._coll._coll_df[self._coll.col_for(7)].sum()
            * 100
        )

    def _np_bndrs_gt_dmns(self):
        self._coll.save_stat([8, 6])
        self._agg_df["np_b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(6)].sum()
            / self._coll._coll_df[self._coll.col_for(8)].sum()
            * 100
        )

    def _num_bndrs(self):
        self._coll.save_stat([0, 7, 8])
        self._agg_df["num_b"] = self._coll._coll_df[self._coll.col_for(0)].sum()
        self._agg_df["num_p_b"] = self._coll._coll_df[self._coll.col_for(7)].sum()
        self._agg_df["num_np_b"] = self._coll._coll_df[self._coll.col_for(8)].sum()

    def save_stat(self):
        self._num_bndrs()
        self._bndrs_gt_dmns()
        self._p_bndrs_gt_dmns()
        self._np_bndrs_gt_dmns()
        self._agg_df["res"] = np.full((len(self._agg_df),), self._coll._res)
        self._agg_df["lim"] = np.full((len(self._agg_df),), self._coll._lim)
        self._agg_df["model"] = np.full(
            (len(self._agg_df),), str(self._coll._prediction)
        )

        return FileSave.append_tsv(
            self._agg_df,
            f"{PathObtain.data_dir()}/generated_data/mcdomains/aggr_mcdmns_stat.tsv",
        )
