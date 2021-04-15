#
# \PicoDst reader for Python
#
# \author Skipper Kagamaster
# \date 03/19/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
# /

# Not all these are used right now.
import numpy as np
import uproot as up
import awkward as ak

# Speed of light, in m/s
SPEED_OF_LIGHT = 299792458
# Proton mass, in GeV
PROTON_MASS = 0.9382720813


def index_cut(a, *args):
    for arg in args:
        arg = arg[a]
        yield arg


# TODO Check to see that these results are reasonable. Graph against p_t.
def rapidity(p_z):
    e_p = np.power(np.add(PROTON_MASS**2, np.power(p_z, 2)), 1/2)
    e_m = np.subtract(PROTON_MASS**2, np.power(p_z, 2))
    e_m = ak.where(e_m < 0.0, 0.0, e_m)  # to avoid imaginary numbers
    e_m = np.power(e_m, 1/2)
    e_m = ak.where(e_m == 0.0, 1e-10, e_m)  # to avoid infinities
    y = np.multiply(np.log(np.divide(e_p, e_m)), 1/2)
    return y


class EPD_Hits:
    mID = None
    mQT_data = None
    mnMip = None

    position = None          # Supersector position on wheel [1, 12]
    tile = None              # Tile number on the Supersector [1, 31]
    row = None               # Row Number [1, 16]
    EW = None                # -1 for East wheel, +1 for West wheel
    ADC = None               # ADC Value reported by QT board [0, 4095]
    TAC = None               # TAC value reported by QT board[0, 4095]
    TDC = None               # TDC value reported by QT board[0, 32]
    has_TAC = None           # channel has a TAC
    nMip = None              # gain calibrated signal, energy loss in terms of MPV of Landau convolution for a MIP
    status_is_good = None    # good status, according to database

    def __init__(self, mID, mQT_data, mnMips, lower_bound=0.2, upper_bound=3):
        self.mID = mID
        self.mQT_data = mQT_data
        self.mnMip = mnMips

        self.has_TAC = np.bitwise_and(np.right_shift(self.mQT_data, 29), 0x1)
        self.status_is_good = np.bitwise_and(np.right_shift(self.mQT_data, 30),  0x1)

        self.adc = np.bitwise_and(self.mQT_data, 0x0FFF)
        self.tac = np.bitwise_and(np.right_shift(self.mQT_data, 12), 0x0FFF)
        self.TDC = np.bitwise_and(np.right_shift(self.mQT_data, 24), 0x001F)

        self.EW = np.sign(self.mID)
        self.position = np.abs(self.mID // 100)
        self.tile = np.abs(self.mID) % 100
        self.row = np.abs(mID) % 100 // 2 + 1
        self.nMip = ak.where(self.status_is_good, self.mnMip, 0)
        # nMIP truncation
        self.nMip = ak.where(self.nMip <= lower_bound, lower_bound, self.nMip)
        self.nMip = ak.where(self.nMip >= upper_bound, upper_bound, self.nMip)

    def generate_epd_hit_matrix(self):
        ring_sum = np.zeros((32, len(self.nMip)))
        print("Filling array of dimension", ring_sum.shape)
        for i in range(32):
            x = i%16
            if i < 16:
                ew_mask = ak.where(self.EW > 0, 1, 0)
            else:
                ew_mask = ak.where(self.EW < 0, 1, 0)
            ring_i = ak.sum(ak.where(self.row == x+1, self.nMip, 0)*ew_mask, axis=-1)
            ring_sum[i] = ring_i
        return ring_sum


class PicoDST:
    """This class makes the PicoDST from the root file, along with
    all of the observables I use for proton kurtosis analysis."""

    def __init__(self, data_file=None):
        """This defines the variables we'll be using
        in the class."""
        self.data: bool
        self.num_events = None
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
        self.epd_hits = None

        if data_file is not None:
            self.import_data(data_file)

    def import_data(self, data_in):
        """This imports the data. You must have the latest versions
        of uproot and awkward installed on your machine (uproot4 and
        awkward 1.0 as of the time of this writing).
        Use pip install uproot awkward.
        Args:
            data_in (str): The path to the picoDst ROOT file"""
        try:
            data = up.open(data_in)["PicoDst"]
            self.num_events = len(data["Event"]["Event.mPrimaryVertexX"].array())
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

            # Load EPD Data
            # I am worried about flattening this data, I need to understand the structure better to figure out how to keep events associated properly
            epd_hit_id_data = data["EpdHit"]["EpdHit.mId"].array()
            epd_hit_mQTdata = data["EpdHit"]["EpdHit.mQTdata"].array()
            epd_hit_mnMIP   = data["EpdHit"]["EpdHit.mnMIP"].array()
            self.epd_hits = EPD_Hits(epd_hit_id_data, epd_hit_mQTdata, epd_hit_mnMIP)

            # print("PicoDst " + data_in[-13:-5] + " loaded.")

        # except ValueError:  # Skip empty picos.
        #     print("ValueError at: " + data_in)  # Identifies the misbehaving file.
        except KeyError:  # Skip non empty picos that have no data.
            print("KeyError at: " + data_in)  # Identifies the misbehaving file.

    def vertex_cuts(self, v_r_cut=2.0, v_z_cut=30.0):
        index = ((np.absolute(self.v_z) <= v_z_cut) & (self.v_r <= v_r_cut))
        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, \
            self.tofmatch, self.bete_eta_1, self.p_t, self.p_g, self.phi, self.dca, \
            self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, self.charge, self.beta, \
            self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid, self.epd_hits.nMip, self.epd_hits.row = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3,
                      self.tofmult, self.tofmatch, self.bete_eta_1, self.p_t, self.p_g,
                      self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.m_2, self.charge, self.beta, self.dedx, self.zdcx,
                      self.rapidity, self.nhitsmax, self.nsigma_proton, self.tofpid,
                      self.epd_hits.nMip, self.epd_hits.row)

    def refmult_correlation_cuts(self, tofmult_refmult=np.array([[2.536, 200], [1.352, -54.08]]),
                                 tofmatch_refmult=np.array([0.239, -14.34]),
                                 beta_refmult=np.array([0.447, -17.88])):
        index = ((self.tofmult <= (np.multiply(tofmult_refmult[0][0], self.refmult3) + tofmult_refmult[0][1])) &
                 (self.tofmult >= (np.multiply(tofmult_refmult[1][0], self.refmult3) + tofmult_refmult[1][1])) &
                 (self.tofmatch >= (np.multiply(tofmatch_refmult[0], self.refmult3) + tofmatch_refmult[1])) &
                 (self.bete_eta_1 >= (np.multiply(beta_refmult[0], self.refmult3) + beta_refmult[1])))
        self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3, self.tofmult, \
            self.tofmatch, self.bete_eta_1, self.p_t, self.p_g, self.phi, self.dca, \
            self.eta, self.nhitsfit, self.nhitsdedx, self.m_2, self.charge, self.beta, \
            self.dedx, self.zdcx, self.rapidity, self.nhitsmax, self.nsigma_proton, \
            self.tofpid, self.epd_hits.nMip, self.epd_hits.row = \
            index_cut(index, self.v_x, self.v_y, self.v_z, self.v_r, self.refmult3,
                      self.tofmult, self.tofmatch, self.bete_eta_1, self.p_t, self.p_g,
                      self.phi, self.dca, self.eta, self.nhitsfit, self.nhitsdedx,
                      self.m_2, self.charge, self.beta, self.dedx, self.zdcx,
                      self.rapidity, self.nhitsmax, self.nsigma_proton, self.tofpid,
                      self.epd_hits.nMip, self.epd_hits.row)
        

class Event_Cuts:
    """
    Event_Cuts takes an object of events and applys the criteria function to it 
    to determin acceptable events.  They can then be accessed in the same way as
    a typical Pico_DST object.

    The criteria function must accept a object of events and the event index as the parameters.
    A typical function might be:
    
    def basic_filter_1cm(events, index):
        return events.v_r[index] < 1  # Only get events within 1 cm of center
    """
    def __init__(self, events, criteria=None, mask=None):
        self.events = events
        self.mask = mask    # Sometimes we want to use the same mask
        if self.mask is None:
            self.mask = self.generate_mask(criteria)
        self.num_events = int(np.sum(self.mask))
        # print(self.mask)
        
    def generate_mask(self, criteria, mask=None):
        """
        generate_mask applies the criteria function to each event to determin if it is acceptable
        """
        if mask is None:
            mask = np.zeros(self.events.num_events, dtype=np.bool)
        for i in range(self.events.num_events):
            if criteria(self.events, i):
                mask[i] = True
        return mask

    def __getattr__(self, name):
        if name == "epd_hits":  # Create an new event cut object for EPD data
            return Event_Cuts(self.events.epd_hits, mask=self.mask)
        array = getattr(self.events, name)[self.mask]   # Return a filterd array, a new copy
        return array
