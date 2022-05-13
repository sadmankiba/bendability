import sys

import numpy as np
import pandas as pd
import logomaker as lm

from models.prediction import Prediction
from util.constants import FigSubDir
from util.util import FileSave


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
        logo = lm.Logo(df, font_name='Arial Rounded MT Bold',)
        logo.ax.set_ylim((0, 2.85))
        if format_ == 'png':
            FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.png")
        else:
            FileSave.figure_in_figdir(f"{FigSubDir.LOGOS}/logo_{str(i)}.svg")