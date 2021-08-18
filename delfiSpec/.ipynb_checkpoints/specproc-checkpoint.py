# Import dependencies
import numpy as np
from apogee.spec import continuum
from apogee.tools import bitmask as bm

from .util import get_DR_slice, bitsNotSet

# Future: Define a class for spectra - spectra, error and weight

def process_spectra(spectra_info=None, badcombpixmask=4351, minSNR=50.):
    cont_cannon = continuum.fit(spectra_info[:, 0], spectra_info[:, 1], type='cannon')

    spectra_info[:, 0] = spectra_info[:, 0]/cont_cannon
    spectra_info[:, 1] = spectra_info[:, 1]/cont_cannon

    # Get DR indices
    spec = get_DR_slice(spectra_info[:, 0])
    spec_err = get_DR_slice(spectra_info[:, 1])
    bitmask = (get_DR_slice(spectra_info[:, 2])).astype('int')

    maskbits = bm.bits_set(badcombpixmask)
    # Mask where SNR low or where something flagged in bitmask
    mask = (spec/spec_err < minSNR) | bitsNotSet(bitmask, maskbits)

    # Errors below 0.005 in APOGEE are not trusted 
    spec_err[spec_err<0.005] = 0.005
    
    try:
        weight = 1.0 / spec_err**2
    except:
        # Handling zero errors
        zero_errors = np.where(spec_err==0)
        listOfCoordinates= list(zip(zero_errors[0], zero_errors[1]))
        for cord in listOfCoordinates:
            spec_err[cord] = np.median(spec_err)

        weight = 1.0 / spec_err**2

    np.place(weight, mask, 0)
    return np.squeeze(spec), np.squeeze(spec_err), np.squeeze(weight)