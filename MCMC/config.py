# -*- coding: utf-8 -*-

import os
import numpy as np

class ConfigHolder:
    """
    Class to hold configuration variables for a generic MCMC sampler
    """
    def __init__(self, save_chain=False):
        self.save_chain = save_chain
