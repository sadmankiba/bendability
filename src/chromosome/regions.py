from __future__ import annotations
from typing import Sequence, Any

import pandas as pd
import numpy as np
from nptyping import NDArray

from util.custom_types import PosOneIdx

regions = pd.DataFrame

START = "start"
END = "end"


def contains(
    containers: regions[START:int, END:int], bps: Sequence[PosOneIdx]
) -> NDArray[(Any,), bool]:
    def _contains_bps(region: pd.Series):
        cntns = False
        for bp in bps:
            if region[START] <= bp <= region[END]:
                cntns = True 
        
        return cntns 
    
    return containers.apply(lambda region: _contains_bps(region), axis = 1).to_numpy()