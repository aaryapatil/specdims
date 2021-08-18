# Import Python Standard Libraries
import os

# Import dependencies
import numpy as np
from astropy.io import fits
import apogee.tools.read as apread
from apogee.tools import bitmask as bm
from apogee.tools import apStarInds
from tqdm import tqdm


"""
Utility functions for pre-processing APOGEE spectra.
"""


def get_DR_slice(arr, DR=12):
    '''
    Get APOGEE Data Release slice on aspcapStarWavegrid.
    
    Parameters
    ----------
    arr : ndarray
          Input array.
    dr  : int
          Data release whose pixel bounds are to be used.
    
    Returns
    -------
    out: ndarray
         Sliced array.    
    '''
    
    inds = []
    # Get the starting and ending pixel indices on the
    # aspcapStarWavegrid for the three APOGEE filters
    for filt, index in apStarInds[str(DR)].items():
        inds.append(index[0])
        inds.append(index[1])

    # Slice at the three filters
    return arr[..., np.r_[inds[0]:inds[1], inds[2]:inds[3], inds[4]:inds[5]]]


def bitsNotSet(bitmask, maskbits):
    """
    Given a bitmask, returns True where any of maskbits are set 
    and False otherwise.
    
    Parameters
    ----------
    bitmask  : ndarray
               Input bitmask.
    maskbits : ndarray
               Bits to check if set in the bitmask
    """
    
    goodLocs_bool = np.zeros(bitmask.shape).astype(bool)
    for m in maskbits:
        bitind = bm.bit_set(m, bitmask)
        goodLocs_bool[bitind] = True
    return goodLocs_bool


class ApogeeCat():
    
    def __init__(self, apogee_cat=None):
        """
        Read spectra from the APOGEE allStar catalogue based on selection criteria.
        
        Parameters
        ----------
        apogee_cat : ndarray, optional
                     APOGEE catalogue of stars. Defaults to the allStar catalogue.
        """
        
        if apogee_cat is None:
            self.apogee_cat = apread.allStar(rmcommissioning=True, rmdups=True)
        else:
            self.apogee_cat = apogee_cat

    def read_OCCAM_cluster(self, cluster='NGC 2682'):
        '''
        Read APOGEE giant and dwarf data for the OCCAM cluster members.
        
        Parameters
        ----------
        cluster : str
                  Name of cluster to read from OCCAM.
        '''
        
        dir_path = os.path.dirname(os.path.realpath(__file__))

        occam_clusters = fits.open(os.path.join(dir_path, '/geir_data/scr/patil/delfispec/data/occam_cluster-DR14.fits'))[1].data
        occam_members = fits.open(os.path.join(dir_path, '/geir_data/scr/patil/delfispec/data/occam_member-DR14.fits'))[1].data
    
        names = np.unique(occam_members['CLUSTER'])
        if cluster in names:
            DM_members = occam_members[np.where((occam_members['CLUSTER']==cluster) & (occam_members['MEMBER_FLAG']=='DM'))]
            GM_members = occam_members[np.where((occam_members['CLUSTER']==cluster) & (occam_members['MEMBER_FLAG']=='GM'))]
            self.cluster_dat = occam_clusters[np.where(occam_clusters['NAME']==(lambda c: '_'.join(c.split(' ')))(cluster))]
    
        DM_apoids = np.in1d((self.apogee_cat['APOGEE_ID']).astype('U100'), DM_members['APOGEE_ID'])
        self.DM_apogee = self.apogee_cat[DM_apoids]
    
        GM_apoids = np.in1d((self.apogee_cat['APOGEE_ID']).astype('U100'), GM_members['APOGEE_ID'])
        self.GM_apogee = self.apogee_cat[GM_apoids]

        return self.DM_apogee, self.GM_apogee

    def read_allStar_spectra(self, apogee_cat_cut=None, X_H='FE_H', limits=[-0.1, 0.1]):
        '''
        Read spectra from the APOGEE catalogue given selection cuts.
        
        Parameters
        ----------
        apogee_cat_cut : ndarray
                         APOGEE catalogue with selection cuts. 
        X_H    : str
                 Abundance to use for applying cuts to APOGEE allStar catalogue.
        limits : array_like
                 X_H abundance limits.
        '''
        
        if apogee_cat_cut is None:
            apogee_cat_cut = self.apogee_cat[(self.apogee_cat[abundance] > limits[0]) & (self.apogee_cat[abundance] < limits[1])]

        spectra_info = np.zeros(shape=(len(apogee_cat_cut), 3, 8575), dtype=np.float32)
    
        for idx, star in tqdm(enumerate(apogee_cat_cut)):
            try:
                spec = apread.apStar(star['LOCATION_ID'], star['APOGEE_ID'].astype('U100'), ext=1, header=False)
                if spec.ndim > 1:
                    spectra_info[idx, 0] = spec[1]
                    spectra_info[idx, 1] = apread.apStar(star['LOCATION_ID'], star['APOGEE_ID'].astype('U100'), ext=2, header=False)[1]
                    spectra_info[idx, 2] = apread.apStar(star['LOCATION_ID'], star['APOGEE_ID'].astype('U100'), ext=3, header=False)[1]
                else:
                    spectra_info[idx, 0] = spec
                    spectra_info[idx, 1] = apread.apStar(star['LOCATION_ID'], star['APOGEE_ID'].astype('U100'), ext=2, header=False)
                    spectra_info[idx, 2] = apread.apStar(star['LOCATION_ID'], star['APOGEE_ID'].astype('U100'), ext=3, header=False)
            except:
                print('Error: Index {} APOGEE_ID {}'.format(idx, star['APOGEE_ID']))
                traceback.print_exc()

        return spectra_info