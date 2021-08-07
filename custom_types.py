from typing import Literal, Union

LIBRARY_NAMES = Literal['cnl', 'rl', 'tl', 'chrvl', 'libl']

YeastChrNum = Literal['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                      'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']

ChrId = Union[YeastChrNum, Literal['VL']]
