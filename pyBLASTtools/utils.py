from scipy import interpolate
import numpy as np

def change_sampling_rate(field1, field2, fs1, fs2, interpolation_kind='linear'):

    '''
    Function to change the sampling of field1 to the same of field2.
    - fs1: sampling rate of field1
    - fs2: sampling rate of field2
    '''

    f = interpolate.interp1d(np.arange(len(field1))/fs1, field1, \
                             kind=interpolation_kind, fill_value='extrapolate')
                    
    field1_new = f(np.arange(len(field2))/fs2)

    assert len(field1_new) == len(field2)

    return field1_new

def remove_drift(field, field_ref, f, f_ref):

    '''
    Function to remove the drift from the a particular field. 
    It uses a reference field, field_ref, and the sampling frequencies 
    for both fields
    '''

    poly = np.polyfit(np.arange(len(field))/f, field, deg=1)
    poly_coeff = np.poly1d(poly)
    field_fit = poly_coeff(np.arange(len(field))/f)

    if f != f_ref:
        field_ref = change_sampling_rate(field_ref, field, f_ref, f)
        f_ref = f

    poly_ref = np.polyfit(np.arange(len(field_ref))/f_ref, field_ref, deg=1)

    poly_mixed = np.array([poly_ref[0], poly[1]])
    poly_coeff_mixed = np.poly1d(poly_mixed)
    field_fit_mixed = poly_coeff_mixed(np.arange(len(field))/f)

    return field-field_fit+field_fit_mixed
