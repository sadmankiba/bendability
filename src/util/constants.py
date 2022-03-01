class GDataSubDir:
    PROMOTERS = "promoters"
    TEST = "test"
    HELSEP = "helical_separation"
    ML_MODEL = "ml_model"
    MOTIF = "motif"


class FigSubDir:
    PROMOTERS = "promoters"
    TEST = "test"
    LINKERS = "linkers"
    CROSSREGIONS = "crossregions"
    BOUNDARIES = "boundaries"
    NDRS = "ndrs"
    CHROMOSOME = "chromosome"


YeastChrNumList = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
)

ChrIdList = YeastChrNumList + ("VL",)

CNL = "cnl"
RL = "rl"
TL = "tl"
CHRVL = "chrvl"
LIBL = "libl"

CNL_LEN = 19907
RL_LEN = 12472
TL_LEN = 82368
CHRVL_LEN = 82404
LIBL_LEN = 92918

SEQ_LEN = 50
ONE_INDEX_START = 1

CHRV_TOTAL_BP = 576871
CHRV_TOTAL_BP_ORIGINAL = 576874
