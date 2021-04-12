def user_defined_lnprior(p):
    '''
    Takes a vector of stellar parameters and returns the ln prior.
    '''
    if not ((p[0] > Teff_lo) and (p[0] < Teff_hi) and
            (p[1] > logg_lo) and (p[1] < logg_hi) and
            (p[2] > vz_lo) and (p[2] < vz_hi) and
            (p[3] > vsini_lo) and (p[3] < vsini_hi) and
            (p[4] > logOmega_lo) and (p[4] < logOmega_hi) and
            (p[8] > sigAmp_lo) and (p[8] < sigAmp_hi) and
            (p[9] > logAmp_lo) and (p[9] < logAmp_hi) and
            (p[10] > ll_lo) and (p[10] < ll_hi) ):
      return -np.inf

    ln_prior_out = 0

    return ln_prior_out
