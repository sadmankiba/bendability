from __future__ import annotations
from pathlib import Path

from matplotlib import colors
import pandas as pd
import numpy as np
from nptyping import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest
import cv2

from chromosome.regions import Regions
from util.util import FileSave, PathObtain
from util.constants import CHRV_TOTAL_BP, CHRV_TOTAL_BP_ORIGINAL, GDataSubDir, FigSubDir
from util.kmer import KMer


N_MOTIFS = 256
LEN_MOTIF = 8


class MotifsM35:
    def __init__(self) -> None:
        self._V = 2
        self._running_score = self._read_running_score()

    def _read_running_score(self) -> NDArray[(N_MOTIFS, CHRV_TOTAL_BP)]:
        mdirs = {1: "model35_parameters_parameter_274_alt", 2: "motif_matching_score"}

        def _score_file(i: int):
            return (
                f"{PathObtain.gen_data_dir()}/{GDataSubDir.MOTIF}/"
                f"{mdirs[self._V]}/motif_{i}"
            )

        scores = np.empty((N_MOTIFS, CHRV_TOTAL_BP))
        for i in range(N_MOTIFS):
            df = pd.read_csv(_score_file(i), header=None)
            assert len(df) == CHRV_TOTAL_BP_ORIGINAL
            scores[i] = df.to_numpy().flatten()[
                : CHRV_TOTAL_BP - CHRV_TOTAL_BP_ORIGINAL
            ]

        return scores

    def enrichment(self, regions: Regions, subdir: str) -> Path:
        enr = self._running_score[:, regions.cover_mask]
        fig, axes = plt.subplots(8, sharey=True)
        fig.suptitle("Motif enrichments")
        for i in range(8):
            axes[i].boxplot(enr[i * 32 : (i + 1) * 32].T, showfliers=True)

        return FileSave.figure_in_figdir(
            f"{subdir}/motif_m35/enrichment_{regions}_{regions.chrm}_v{self._V}.png"
        )

    def enrichment_compare(self, rega: Regions, regb: Regions, subdir: str):
        enra = self._running_score[:, rega.cover_mask]
        enrb = self._running_score[:, regb.cover_mask]
        z = [(i,) + ztest(enra[i], enrb[i]) for i in range(N_MOTIFS)]
        df = pd.DataFrame(z, columns=["motif_no", "ztest_val", "p_val"])
        df["ztest_val"] = -df["ztest_val"]
        
        fn = f"enrichment_comp_{rega}_{regb}_{rega.chrm}_v{self._V}"
        FileSave.tsv_gdatadir(
            df,
            f"{subdir}/motif_m35/{fn}.tsv",
        )
        FileSave.tsv_gdatadir(
            df.sort_values("ztest_val"),
            f"{subdir}/motif_m35/sorted_{fn}.tsv",
        )


class PlotMotifs:
    @classmethod
    def integrate_logos(cls) -> Path:
        ztest_str = {
            1: "res_200_lim_100_perc_0.5_fanc_domains_res_200_lim_100_perc_0.5_fanc",
            2: "bfn_lnk_0_bf_res_200_lim_100_perc_0.5_fanc_dmnsfn_bfn_lnk_0_bf_res_200_lim_100_perc_0.5_fanc_chrm_s_mcvr_m_None_VL_v2",
        }
        sel = 2
        dir = f"{PathObtain.figure_dir()}/{FigSubDir.MOTIFS}"
        imrows = []

        score_df = pd.read_csv(
            f"{PathObtain.gen_data_dir()}/boundaries/motif_m35/enrichment_comp_{ztest_str[sel]}.tsv",
            sep="\t",
        )
        score_df = score_df.sort_values(by="ztest_val", ignore_index=True)
        for i in range(16):
            row = []
            for j in range(16):
                n, z, p = tuple(score_df.loc[i * 16 + j])
                logo = cv2.imread(
                    f"{dir}//model35_parameters_parameter_274_merged_motif/{int(n)}.png"
                )
                z, p = round(z, 2), round(p, 2)
                logo = cls._add_score(logo, z, p)
                row.append(logo)
            imrows.append(cv2.hconcat(row))

        img = cv2.vconcat(imrows)
        impath = Path(f"{dir}/integrated_z_score_{ztest_str[sel]}.png")
        cv2.imwrite(str(impath), img)
        return impath

    @classmethod
    def _add_score(cls, img: np.ndarray, z: float, p: float):
        new_img = np.ascontiguousarray(
            np.vstack([np.full((30, img.shape[1], 3), fill_value=255), img]),
            dtype=np.uint8,
        )

        pos = (10, 28)
        font_scale = 1.1
        font_color = (0, 0, 0)
        thickness = 1
        linetype = 2
        cv2.putText(
            new_img,
            f"z={z}, p={p}",
            pos,
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            font_color,
            thickness,
            linetype,
        )
        return new_img


class KMerMotifs:
    @classmethod
    def score(cls, rega: Regions, regb: Regions, subdir: str):
        asq = np.array(list(rega.chrm.seq))[rega.cover_mask]
        bsq = np.array(list(regb.chrm.seq))[regb.cover_mask]

        def _c(kmer: str):
            ac = KMer.count(kmer, "".join(asq.tolist())) / len(asq)
            bc = KMer.count(kmer, "".join(bsq.tolist())) / len(bsq)
            return kmer, ac, bc

        n = 4
        df = pd.DataFrame(
            list(map(lambda s: _c(s), KMer.all(n))), columns=["kmer", "a_cnt", "b_cnt"]
        )
        df["diff"] = df["a_cnt"] - df["b_cnt"]

        fn = f"kmer{n}_motif_{rega}_{regb}_{rega.chrm}"
        FileSave.tsv_gdatadir(df, f"{subdir}/{fn}.tsv")
        FileSave.tsv_gdatadir(
            df.sort_values("diff"),
            f"{subdir}/sorted_{fn}.tsv",
        )


MOTIF_ID = "motif_id"
TF = "tf"
CONTRIB_SCORE = "contrib_score"


class MotifsM30:
    def __init__(self):
        self._tomtom_motif_file = (
            f"{PathObtain.input_dir()}/motifs/tomtom_model30_yeastract.tsv"
        )
        self._contrib_score_file = (
            f"{PathObtain.input_dir()}/motifs/contribution_scores_model30_train9.txt"
        )
        self._BEST_MATCH_PERC = 25
        self._motifs = self._read_tomtom_motifs()
        self._contrib_scores = self._read_contrib_scores()

    def plot_ranked_tf(self):
        tfdf = self.ranked_tf()
        tfdf = tfdf.drop(np.arange(10, len(tfdf) - 10))
        mpl.rcParams.update({"font.size": 18})
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Move bottim x-axis to centre
        ax.spines["bottom"].set_position(("data", 0))

        # Eliminate upper and right axes
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        plt.bar(tfdf[TF], tfdf[CONTRIB_SCORE])
        plt.ylabel("Average Contribution Score")
        plt.setp(ax.get_xticklabels(), rotation=90, va="top")
        return FileSave.figure_in_figdir(
            f"motifs/ranked_tf_perc_{self._BEST_MATCH_PERC}.png"
        )

    def ranked_tf(self) -> pd.DataFrame:
        mdf = pd.merge(
            self._motifs, self._contrib_scores, how="left", on=MOTIF_ID
        ).drop([MOTIF_ID], axis=1)

        return mdf.groupby(["tf"]).mean().sort_values(by=CONTRIB_SCORE).reset_index()

    def _read_tomtom_motifs(self) -> pd.DataFrame[MOTIF_ID:int, TF:str]:
        df = pd.read_csv(
            self._tomtom_motif_file,
            sep="\t",
            usecols=["Query_ID", "Target_ID", "p-value"],
        )
        df = df.iloc[:-3]  # Remove comments

        df = df.sort_values(by="p-value").iloc[
            : int(len(df) * self._BEST_MATCH_PERC / 100)
        ]

        df[MOTIF_ID] = df.apply(lambda row: int(row["Query_ID"].split("-")[1]), axis=1)
        df[TF] = df.apply(
            lambda row: row["Target_ID"].split("&")[0][:4].upper(), axis=1
        )
        return df.drop(["Query_ID", "Target_ID", "p-value"], axis=1)

    def _read_contrib_scores(
        self,
    ) -> pd.DataFrame[MOTIF_ID:int, CONTRIB_SCORE:float]:
        df = pd.read_csv(self._contrib_score_file, names=[CONTRIB_SCORE])
        df[MOTIF_ID] = np.arange(256)
        return df

    def sorted_contrib(self) -> list[int]:
        return self._contrib_scores.sort_values(CONTRIB_SCORE)[MOTIF_ID].to_list()
