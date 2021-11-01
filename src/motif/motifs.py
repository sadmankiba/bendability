from __future__ import annotations
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from util.util import FileSave, PathObtain

MOTIF_ID = "motif_id"
TF = "tf"
CONTRIB_SCORE = "contrib_score"


class Motifs:
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
