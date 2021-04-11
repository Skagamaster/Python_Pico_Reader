#
# \PicoDst reader for Python
#
# \author Skipper KAgamaster
# \date 03/19/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

# Not all these are used right now.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema as arex
from scipy.signal import savgol_filter as sgf
from scipy.stats import skew, kurtosis
import uproot as up
import awkward as ak
import time
import logging

# Speed of light, in m/s
c = 299792458
# Proton mass, in GeV
mp = 0.9382720813


def index_cut(a, *args):
    for arg in args:
        arg = arg[a]
        yield arg


# TODO Check to see that these results are reasonable. Graph against p_t.
def rapidity(p_z):
    e_p = np.power(np.add(mp**2, np.power(p_z, 2)), 1/2)
    e_m = np.subtract(mp**2, np.power(p_z, 2))
    e_m = ak.where(e_m < 0.0, 0.0, e_m)  # to avoid imaginary numbers
    e_m = np.power(e_m, 1/2)
    e_m = ak.where(e_m == 0.0, 1e-10, e_m)  # to avoid infinities
    y = np.multiply(np.log(np.divide(e_p, e_m)), 1/2)
    return y


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all of the observables I use for proton kurtosis analysis."""

    def __init__(self):
        """This defines the variables we'll be using
        in the class."""
        self.data: bool
        self.v_x = None
        self.v_y = None
        self.v_z = None
        self.v_r = None
        self.refmult3 = None
        self.tofmult = None
        self.tofmatch = None
        self.bete_eta_1 = None
        self.p_t = None
        self.p_g = None
        self.phi = None
        self.dca = None
        self.eta = None
        self.nhitsfit = None
        self.nhitsdedx = None
        self.m_2 = None
        self.charge = None
        self.beta = None
        self.dedx = None
        self.zdcx = None
        self.rapidity = None
        self.nhitsmax = None
        self.nsigma_proton = None
        self.tofpid = None
        self.protons = None
        self.antiprotons = None
        self.dedx_histo = None
        self.p_g_histo = None
        self.charge_histo = None

    def import_data(self, data_in):
        """This imports the data. You must have the latest versions
        of uproot and awkward installed on your machine (uproot4 and
        awkward 1.0 as of the time of this writing).
        Use pip install uproot awkward.
        Args:
            data_in (str): The path to the picoDst ROOT file"""
        try:
            data = up.open(data_in)["PicoDst"]
            # Make vertices
            self.v_x = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexX"].array()))
            self.v_y = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexY"].array()))
            self.v_z = ak.to_numpy(ak.flatten(data["Event"]["Event.mPrimaryVertexZ"].array()))
            self.v_r = np.sqrt(np.power(np.subtract(np.mean(self.v_x), self.v_x), 2) +
                               np.power(np.subtract(np.mean(self.v_y), self.v_y), 2))
            self.zdcx = ak.to_numpy(ak.flatten(data["Event"]["Event.mZDCx"].array()))
            self.refmult3 = ak.to_numpy(ak.flatten(data["Event"]["Event.mRefMult3PosEast"].array() +
                                                   data["Event"]["Event.mRefMult3PosWest"].array() +
                                                   data["Event"]["Event.mRefMult3NegEast"].array() +
                                                   data["Event"]["Event.mRefMult3NegWest"].array()))
            self.tofmult = ak.to_numpy(ak.flatten(data["Event"]["Event.mbTofTrayMultiplicity"].array()))
            self.tofmatch = ak.to_numpy(ak.flatten(data["Event"]["Event.mNBTOFMatch"].array()))
            # Make p_g and p_t
            p_x = data["Track"]["Track.mGMomentumX"].array()
            p_y = data["Track"]["Track.mGMomentumY"].array()
            p_y = ak.where(p_y == 0.0, 1e-10, p_y)  # to avoid infinities
            p_z = data["Track"]["Track.mGMomentumZ"].array()
            self.p_t = np.sqrt(np.power(p_x, 2) + np.power(p_y, 2))
            self.p_g = np.sqrt((np.power(p_x, 2) + np.power(p_y, 2) + np.power(p_z, 2)))
            self.eta = np.arcsinh(np.divide(p_z, self.p_t))
            self.rapidity = rapidity(p_z)
            # Make dca
            dca_x = data["Track"]["Track.mOriginX"].array() - self.v_x
            dca_y = data["Track"]["Track.mOriginY"].array() - self.v_y
            dca_z = data["Track"]["Track.mOriginZ"].array() - self.v_z
            self.dca = np.sqrt((np.power(dca_x, 2) + np.power(dca_y, 2) + np.power(dca_z, 2)))
            self.nhitsdedx = data["Track"]["Track.mNHitsDedx"].array()
            self.nhitsfit = data["Track"]["Track.mNHitsFit"].array()
            self.nhitsmax = data["Track"]["Track.mNHitsMax"].array()
            self.nhitsmax = ak.where(self.nhitsmax == 0, 1e-10, self.nhitsmax)  # to avoid infinities
            self.dedx = data["Track"]["Track.mDedx"].array()
            self.nsigma_proton = data["Track"]["Track.mNSigmaProton"].array()
            self.charge = ak.where(self.nhitsfit >= 0, 1, -1)
            self.beta = data["BTofPidTraits"]["BTofPidTraits.mBTofBeta"].array()/20000.0
            self.tofpid = data["BTofPidTraits"]["BTofPidTraits.mTrackIndex"].array()
            # Make B_n_1
            be1_1 = ak.sum(ak.where(self.beta > 0.1, 1, 0), axis=-1)
            be1_2 = ak.sum(ak.where(np.absolute(self.eta) < 1.0, 1, 0), axis=-1)
            be1_3 = ak.sum(ak.where(self.dca < 3.0, 1, 0), axis=-1)
            be1_4 = ak.sum(ak.where(np.absolute(self.nhitsfit) > 10, 1, 0), axis=-1)
            self.bete_eta_1 = be1_1 + be1_2 + be1_3 + be1_4
            # Make m^2
            p_squared = np.power(self.p_g[self.tofpid], 2)
            b_squared = np.power(self.beta, 2)
            b_squared = ak.where(b_squared == 0.0, 1e-10, b_squared)  # to avoid infinities
            g_squared = np.subtract(1, b_squared)
            self.m_2 = np.divide(np.multiply(p_squared, g_squared), b_squared)
            # Make phi.
            o_x = data["Track"]["Track.mOriginX"].array()
            o_y = data["Track"]["Track.mOriginY"].array()
            self.phi = np.arctan2(o_y, o_x)

            # print("PicoDst " + data_in[-13:-5] + " loaded.")

        except ValueError:  # Skip empty picos.
            print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.

    def event_cuts(self, v_r_cut=2.0, v_z_cut=30.0, tofmult_refmult=np.array([[2.536, 200], [1.352, -54.08]]),
                   tofmatch_refmult=np.array([0.239, -14.34]), beta_refmult=np.array([0.447, -17.88])):
        """This is used to make event cuts.
        """
        index = ((np.absolute(self.v_z) <= v_z_cut) & (self.v_r <= v_r_cut) &
                 (self.tofmult <= (np.multiply(tofmult_refmult[0][0], self.refmult3) + tofmult_refmult[0][1])) &
                 (self.tofmult >= (np.multiply(tofmult_refmult[1][0], self.refmult3) + tofmult_refmult[1][1])) &
                 (self.tofmatch >= (np.multiply(tofmatch_refmult[0], self.refmult3) + tofmatch_refmult[1])) &
                 (self.bete_eta_1 >= (np.multiply(beta_refmult[0], self.refmult3) + beta_refmult[1])))

        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, self.tofmatch, self.bete_eta_1, \
            self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, \
            self.charge, self.beta, self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, self.tofmatch,
                      self.bete_eta_1, self.p_t, self.p_g, self.phi, self.dca, self.eta, self.nhitsfit,
                      self.nhitsdedx, self.m_2, self.charge, self.beta, self.dedx, self.zdcx, self.rapidity,
                      self.nhitsmax, self.nsigma_proton, self.tofpid)
