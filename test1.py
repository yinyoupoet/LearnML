# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:30:00 2018

@author: hasee
"""

import numpy as np
from sklearn import preprocessing

data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])

data_standardized = preprocessing.scale(data)
