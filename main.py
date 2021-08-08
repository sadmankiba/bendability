from prediction import Prediction
from chromosome import ChrIdList, YeastChrNumList
from loops import MultiChrmMeanLoopsCollector

if __name__ == '__main__':
    MultiChrmMeanLoopsCollector(Prediction(30), ('VL',)).plot_loop_cover_frac()
