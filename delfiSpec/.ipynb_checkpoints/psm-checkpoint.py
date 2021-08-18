import numpy as np

## Polynomial Model from Yuan-Sen Ting (Rix+ 2017) ##

psminfo = np.load('/geir_data/scr/patil/psm/kurucz_quadratic_psm.npz')
coeff_array = psminfo['coeff_array']

# a set of training labels
training_labels = psminfo['training_labels']
wavelength = psminfo['wavelength']

# auxiliary arrays to reconstruct the spectrum (because we need to choose a reference point to "Taylor-expand"
inds = psminfo['indices']
reference_flux = psminfo['reference_flux']
reference_point = psminfo['reference_point']
Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,tih,vh,mnh,nih,feh,c12c13 = reference_point

#LABEL ORDER Teff [1000K], logg, vturb [km/s] (micro), ch, nh, oh, nah, mgh, alh, sih, sh, kh, cah, 
#tih, vh, mnh, nih, feh, log10(c12c13)

#==================================================
# make generate APOGEE spectrum
def generate_spectrum(labels=None,Teff=Teff,logg=logg,vturb=vturb,ch=ch,nh=nh,oh=oh,nah=nah,mgh=mgh,
                      alh=alh,sih=sih,sh=sh,kh=kh,cah=cah,tih=tih,vh=vh,mnh=mnh,nih=nih,feh=feh,
                      c12c13=c12c13):
    if not isinstance(labels,(list,np.ndarray)):
        labels = np.array([Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,tih,vh,mnh,nih,
                           feh,c12c13])
    
    # make quadratic labels
    linear_terms = np.array(labels) - reference_point
    quadratic_terms = np.einsum('i,j->ij',linear_terms,linear_terms)\
                            [inds[:,0],inds[:,1]]
    lvec = np.hstack((linear_terms, quadratic_terms))
    
    # generate spectrum
    spec_generate = np.dot(coeff_array,lvec) + reference_flux
    return spec_generate
