import pandas as pd
from numpy import array, median
from hw import Jamshidian as jamsh
from hw import Henrard as henr
from hw import hw_helper
from hw import calibration
from fox_toolbox.utils import xml_parser, rates
from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    

    @abstractmethod
    def calibrate(self, calib_basket):
        pass
    



class HullWhite(Model):
    "Theoretical Formulas for Hull White Model"
    
    def __init__(self, mr=0., sigma=None):
        self.mr = mr
        self.sigma = sigma


    def calibrate(self, calib_basket, dsc_curve, estim_curve, IsJamsh=False):
        calibrationHW = calibration.calibrate_sigma_hw(calib_basket, self.mr, dsc_curve, estim_curve, IsJamsh=False)
        self.sigma = calibrationHW.sigma
        calibrationHW.plot()
        return calibrationHW.data