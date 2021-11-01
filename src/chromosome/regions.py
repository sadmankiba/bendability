from __future__ import annotations
from typing import Iterable, NamedTuple, Sequence, Any

import pandas as pd
import numpy as np
from nptyping import NDArray

from util.custom_types import PosOneIdx

regions = pd.DataFrame

START = "start"
END = "end"

class Regions:
    @classmethod
    def is_in(self, bps: Iterable[PosOneIdx], containers: regions[START:int, END:int]):
        pass

    @classmethod
    def contains(
        self, containers: regions[START:PosOneIdx, END:PosOneIdx], bps: Iterable[PosOneIdx]
    ) -> NDArray[(Any,), bool]:
        def _contains_bps(region: NamedTuple[START: PosOneIdx, END: PosOneIdx]):
            cntns = False
            for bp in bps:
                if getattr(region, START) <= bp <= getattr(region, END):
                    cntns = True 
            
            return cntns 
        
        return np.array(list(map(lambda cnt: _contains_bps(cnt), containers)))