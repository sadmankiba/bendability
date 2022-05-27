import sys

import numpy as np
import pandas as pd
import logomaker as lm
from matplotlib import rc
import matplotlib.pyplot as plt

from models.prediction import Prediction
from util.constants import FigSubDir
from util.util import FileSave, PlotUtil


def gen_motif_logos():
    format_ = "png"

    model = Prediction(35)._model

    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

    museum_layer_num = 2
    museum_layer = model.layers[museum_layer_num]
    _, ic_scaled_prob = museum_layer.get_motifs()

    npa = np.array(ic_scaled_prob)
    for i in range(npa.shape[0]):
        df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T'])
        print(df.head())
        # rc('font', weight='bold')
        # plt.rcParams["font.weight"] = "bold"
        # plt.rcParams["axes.labelweight"] = "bold"
        
        PlotUtil.font_size(36)
        logo = lm.Logo(df, font_name='Arial Rounded MT Bold',)
        logo.ax.set_ylim((0, 2.2))
        logo.ax.set_ylabel("bits")
        logo.ax.set_yticks([0, 2])
        logo.ax.set_xticks(range(0, 8))
        if format_ == 'png':
            FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.png")
        else:
            FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.svg")