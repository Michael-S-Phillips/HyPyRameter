"""Legacy helpers.

The cube functions (``getBand``, ``getBandDepth``, ...) now delegate to
:mod:`hypyrameter.bands` / :mod:`hypyrameter.parameters`; the point-spectrum
helpers (``getRvalue*``) and the .sed / USGS readers are kept as they were.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline as cs

from hypyrameter import parameters as _p
from hypyrameter.bands import band, closest_wavelength
from hypyrameter.browse import stretch


def getClosestWavelength(wl, band_list):
    return closest_wavelength(np.asarray(band_list, dtype=np.float64), wl)


def getBand(cube, wvt, wl, kwidth=5):
    return band(cube, wvt, wl, kwidth)


def getBandDepth(cube, wvt, low, mid, hi, lw=5, mw=5, hw=5):
    return _p.band_depth(_p.Spectra(cube, wvt), low, mid, hi, lw, mw, hw)


def getBandDepthInvert(cube, wvt, low, mid, hi, lw=5, mw=5, hw=5):
    return _p.band_depth_invert(_p.Spectra(cube, wvt), low, mid, hi, lw, mw, hw)


def getBandArea(cube, wvt, low, high, lw=5, hw=5):
    return _p.band_area(_p.Spectra(cube, wvt), low, high, lw, hw)


def getSlope(cube, wvt, low, high, kwidth=5):
    return _p.slope(_p.Spectra(cube, wvt), low, high, kwidth)


def getBandRatio(cube, wvt, num_l, denom_l, num_w=5, denom_w=5):
    return _p.band_ratio(_p.Spectra(cube, wvt), num_l, denom_l, num_w, denom_w)


def getNDI(cube, wvt, a_l, b_l):
    return _p.normalized_difference(_p.Spectra(cube, wvt), a_l, b_l)


def getCubicSplineIntegral(args):
    """Integral of the cubic spline through (x, y); ``args`` is ``(x, y)``."""
    x, y = args
    return _p.spline_area(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def stretchBand(p, stype="linear", perc=2, factor=2.5):
    return stretch(p, stype, perc, factor)


def stretchNBands(img, stype="linear", perc=2, factor=2.5):
    return np.dstack([stretch(img[..., i], stype, perc, factor) for i in range(img.shape[-1])])


def browse2bit(B):
    return np.nan_to_num(B, nan=0.0).astype(np.uint8)


def buildSummary(p1, p2, p3):
    return np.dstack((p1, p2, p3))


# ----------------------------------------------------------------
# point-spectrum helpers and file readers (unchanged from 0.2.x)
# ----------------------------------------------------------------
def getRvalue(spectrum, wvt, wl, kwidth=5):
    """
    Returns the R value of a given spectrum at a specified wavelength.

    Parameters:
    spectrum (pandas.DataFrame): A pandas DataFrame containing the spectrum data.
    wvt (list): A list of wavelengths corresponding to the spectrum data.
    wl (float): The wavelength at which to calculate the R value.
    kwidth (int): The width of the kernel to use for calculating the R value. Default is 5.

    Returns:
    float: The R value of the spectrum at the specified wavelength.
    """
    delta = [q - wl for q in wvt]
    bindex = delta.index(min(delta, key=abs))
    if kwidth == 1:
        r = spectrum.iloc[bindex]
    else:
        w = (kwidth - 1) / 2
        min_index = bindex - w
        max_index = bindex + w
        if bindex - w < 0:
            min_index = 0
        if bindex + w > len(spectrum) - 1:
            max_index = len(spectrum) - 1
        r = np.median(spectrum.iloc[int(min_index) : int(max_index)])
    return r


def getRvalueDepth(spectrum, wvt, low, mid, hi, lw=5, mw=5, hw=5):
    """
    Compute the band depth using precomputed a and b.

    Args:
    spectrum (numpy.ndarray): 1D array of spectral values.
    wvt (numpy.ndarray): 1D array of wavelength values.
    low (float): Lower wavelength value.
    mid (float): Middle wavelength value.
    hi (float): Higher wavelength value.
    lw (int): Width of the lower band.
    mw (int): Width of the middle band.
    hw (int): Width of the higher band.

    Returns:
    float: The computed band depth.
    """
    # retrieve bands from spectrum
    Rlow = getRvalue(spectrum, wvt, low, kwidth=lw)
    Rmid = getRvalue(spectrum, wvt, mid, kwidth=mw)
    Rhi = getRvalue(spectrum, wvt, hi, kwidth=hw)

    # determine wavelengths for low, mid, hi
    WL = getClosestWavelength(low, wvt)
    WM = getClosestWavelength(mid, wvt)
    WH = getClosestWavelength(hi, wvt)

    a = (WM - WL) / (WH - WL)  # a gets multipled by the longer band
    b = 1.0 - a  # b gets multiplied by the shorter band

    # compute the band depth using precomputed a and b
    paramValue = 1.0 - (Rmid / (b * Rlow + a * Rhi))
    return paramValue


def getRvalueDepthInvert(spectrum, wvt, low, mid, hi, lw=5, mw=5, hw=5):
    """
    Computes the band depth using the inverted method.

    Args:
        spectrum (numpy.ndarray): The spectrum to compute the band depth from.
        wvt (numpy.ndarray): The wavelengths of the spectrum.
        low (float): The lower wavelength of the band.
        mid (float): The middle wavelength of the band.
        hi (float): The higher wavelength of the band.
        lw (float, optional): The width of the lower band. Defaults to 5.
        mw (float, optional): The width of the middle band. Defaults to 5.
        hw (float, optional): The width of the higher band. Defaults to 5.

    Returns:
        float: The computed band depth.
    """
    # retrieve bands from spectrum
    Rlow = getRvalue(spectrum, wvt, low, kwidth=lw)
    Rmid = getRvalue(spectrum, wvt, mid, kwidth=mw)
    Rhi = getRvalue(spectrum, wvt, hi, kwidth=hw)

    # determine wavelength values for closest channels
    WL = getClosestWavelength(low, wvt)
    WM = getClosestWavelength(mid, wvt)
    WH = getClosestWavelength(hi, wvt)
    a = (WM - WL) / (WH - WL)  # a gets multipled by the longer band
    b = 1.0 - a  # b gets multiplied by the shorter band

    # compute the band depth using precomputed a and b
    paramValue = 1.0 - ((b * Rlow + a * Rhi) / Rmid)
    if paramValue is -np.inf:
        paramValue = np.nan
    return paramValue


def getRvalueRatio(spectrum, wvt, num_l, denom_l, num_w=5, denom_w=5):
    """
    Calculates the ratio of two R-values obtained from the given spectrum and wavelet transform.

    Args:
        spectrum (numpy.ndarray): The spectrum to calculate R-values from.
        wvt (numpy.ndarray): The wavelet transform to use for calculating R-values.
        num_l (int): The length of the numerator window.
        denom_l (int): The length of the denominator window.
        num_w (int, optional): The width of the numerator window. Defaults to 5.
        denom_w (int, optional): The width of the denominator window. Defaults to 5.

    Returns:
        float: The ratio of the two R-values. If the ratio is -inf, returns NaN instead.
    """
    num = getRvalue(spectrum, wvt, num_l, kwidth=num_w)
    denom = getRvalue(spectrum, wvt, denom_l, kwidth=denom_w)
    paramValue = num / denom
    if paramValue is -np.inf:
        paramValue = np.nan
    return paramValue


def getRvalueArea(spec, wvt, low, high, lw=5, hw=5):
    """retrieve band area

    Args:
        spec (array): pandas df of spectrum
        wvt (list): pandas df of wavelength values
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): low wavelength median filter kernel width. Defaults to 5.
        hw (int, optional): high wavelength median filter kernel width. Defaults to 5.

    Returns:
        (array): band area value calculated in microns by reflectance
    """
    y1 = getRvalue(spec, wvt, low, kwidth=lw)
    delta = [q - low for q in wvt]
    x1 = int(delta.index(min(delta, key=abs)))
    y2 = getRvalue(spec, wvt, high, kwidth=hw)
    delta = [q - high for q in wvt]
    x2 = int(delta.index(min(delta, key=abs)))
    woi_ = np.linspace(
        wvt.iloc[x1], wvt.iloc[x2], int(wvt.iloc[x2] - wvt.iloc[x1] + 1), dtype=int
    ).tolist()
    woi = [wvt.sub(w).abs().idxmin() for w in woi_]
    # remove duplicates from woi
    woi = list(dict.fromkeys(woi))
    wol = [wvt.iloc[w] for w in woi]
    x1 = wol[0]
    x2 = wol[-1]
    m = (y2 - y1) / (x2 - x1)  # m is the slope at all pixels
    b = y2 - m * x2  # b is the intercept at all pixels
    h = [getRvalue(spec, wvt, wvt[i], kwidth=1) - (m * wvt[i] + b) for i in woi]
    ba = -0.001 * getCubicSplineIntegral((wol, h))
    return ba


def getRvalueMin(spec, wvt, low, high, lw=5, hw=5):
    """retrieve minimum value of an absorption band

    Args:
        spec (array): pandas df of spectrum
        wvt (list): pandas df of wavelength values
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): low wavelength median filter kernel width. Defaults to 5.
        hw (int, optional): high wavelength median filter kernel width. Defaults to 5.

    Returns:
        (array): minimum value of an absorption band
    """
    y1 = getRvalue(spec, wvt, low, kwidth=lw)
    delta = [q - low for q in wvt]
    x1 = int(delta.index(min(delta, key=abs)))
    y2 = getRvalue(spec, wvt, high, kwidth=hw)
    delta = [q - high for q in wvt]
    x2 = int(delta.index(min(delta, key=abs)))
    woi_ = np.linspace(
        wvt.iloc[x1], wvt.iloc[x2], int(wvt.iloc[x2] - wvt.iloc[x1] + 1), dtype=int
    ).tolist()
    woi = [wvt.sub(w).abs().idxmin() for w in woi_]
    # remove duplicates from woi
    woi = list(dict.fromkeys(woi))
    wol = [wvt.iloc[w] for w in woi]
    x1 = wol[0]
    x2 = wol[-1]
    m = (y2 - y1) / (x2 - x1)  # m is the slope at all pixels
    b = y2 - m * x2  # b is the intercept at all pixels
    # s0, s1 = y1.shape
    # s2 = len(woi)
    y = m * np.array(wol) + b
    h = [getRvalue(spec, wvt, wvt[i], kwidth=1) - (m * wvt[i] + b) for i in woi]

    # Fit a cubic spline using wol and h
    cs_ = cs(wol, h)  # Find the minimum y value using optimization

    # ----------------------------------------------------------------
    # option 1

    # Generate a finer mesh of x values
    x_finer = np.linspace(min(wol), max(wol), 1000)  # Change 1000 to the desired number of points

    # Calculate y values corresponding to the finer mesh of x values
    y_finer = cs_(x_finer)

    # Find the index of the minimum y value
    min_y_index = np.argmin(y_finer)

    # The x value at the minimum y value
    x_at_min_y = 0.001 * x_finer[min_y_index]
    # ----------------------------------------------------------------
    # option 2

    # Define a function that calculates the value of the CubicSpline at a given x
    # def spline_at_x(x_value):
    #     return cs_(x_value)

    # # Use minimize_scalar to find the x value corresponding to the minimum y
    # result = minimize_scalar(spline_at_x, bounds=(min(wol), max(wol)), method='bounded')

    # # The x value corresponding to the minimum y value
    # x_at_min_y = 0.001*result.x

    return x_at_min_y


def getRvalueFWHM(spec, wvt, low, high, lw=5, hw=5):
    """retrieve minimum value of an absorption band

    Args:
        spec (array): pandas df of spectrum
        wvt (list): pandas df of wavelength values
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): low wavelength median filter kernel width. Defaults to 5.
        hw (int, optional): high wavelength median filter kernel width. Defaults to 5.

    Returns:
        (array): minimum value of an absorption band
    """
    y1 = getRvalue(spec, wvt, low, kwidth=lw)
    delta = [q - low for q in wvt]
    x1 = int(delta.index(min(delta, key=abs)))
    y2 = getRvalue(spec, wvt, high, kwidth=hw)
    delta = [q - high for q in wvt]
    x2 = int(delta.index(min(delta, key=abs)))
    woi_ = np.linspace(
        wvt.iloc[x1], wvt.iloc[x2], int(wvt.iloc[x2] - wvt.iloc[x1] + 1), dtype=int
    ).tolist()
    woi = [wvt.sub(w).abs().idxmin() for w in woi_]
    # remove duplicates from woi
    woi = list(dict.fromkeys(woi))
    wol = [wvt.iloc[w] for w in woi]
    x1 = wol[0]
    x2 = wol[-1]
    m = (y2 - y1) / (x2 - x1)  # m is the slope at all pixels
    b = y2 - m * x2  # b is the intercept at all pixels
    # s0, s1 = y1.shape
    # s2 = len(woi)
    y = m * np.array(wol) + b
    h = [getRvalue(spec, wvt, wvt[i], kwidth=1) - (m * wvt[i] + b) for i in woi]

    # Fit a cubic spline using wol and h
    cs_ = cs(wol, h)  # Find the minimum y value using optimization

    # ----------------------------------------------------------------
    # option 1

    # Generate a finer mesh of x values
    x_finer = np.linspace(min(wol), max(wol), 1000)  # Change 1000 to the desired number of points

    # Calculate y values corresponding to the finer mesh of x values
    y_finer = cs_(x_finer)

    # Find the index of the minimum y value
    min_y_index = np.argmin(y_finer)

    # The x value at the minimum y value
    x_at_min_y = x_finer[min_y_index]
    y_min = y_finer[min_y_index]

    # ----------------------------------------------------------------
    # option 2
    # Define a function that calculates the value of the CubicSpline at a given x
    # def spline_at_x(x_value):
    #     return cs_(x_value)

    # # Use minimize_scalar to find the x value corresponding to the minimum y
    # result = minimize_scalar(spline_at_x, bounds=(min(wol), max(wol)), method='bounded')

    # # The x value corresponding to the minimum y value
    # x_at_min_y = result.x
    # y_min = result.fun

    # calculate the full width half maximum of the band defined by wol and h
    half_max = y_min / 2
    peak_index = np.argmin(np.abs(h - y_min))

    # Find the indices where the intensity is closest to half of the maximum intensity
    try:
        left_index = np.argmin(np.abs(h[:peak_index] - half_max))
        right_index = np.argmin(np.abs(h[peak_index:] - half_max)) + peak_index

        # Calculate the FWHM
        fwhm = 0.001 * (wol[right_index] - wol[left_index])
    except:
        fwhm = np.nan

    return fwhm


def getRvalueAsymmetry(spec, wvt, low, high, lw=5, hw=5):
    """retrieve band asymmetry

    Args:
        spec (array): pandas df of spectrum
        wvt (list): pandas df of wavelength values
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): low wavelength median filter kernel width. Defaults to 5.
        hw (int, optional): high wavelength median filter kernel width. Defaults to 5.

    Returns:
        (array): band asymmetry value
    """
    y1 = getRvalue(spec, wvt, low, kwidth=lw)
    delta = [q - low for q in wvt]
    x1 = int(delta.index(min(delta, key=abs)))
    y2 = getRvalue(spec, wvt, high, kwidth=hw)
    delta = [q - high for q in wvt]
    x2 = int(delta.index(min(delta, key=abs)))
    woi_ = np.linspace(
        wvt.iloc[x1], wvt.iloc[x2], int(wvt.iloc[x2] - wvt.iloc[x1] + 1), dtype=int
    ).tolist()
    woi = [wvt.sub(w).abs().idxmin() for w in woi_]
    # remove duplicates from woi
    woi = list(dict.fromkeys(woi))
    wol = [wvt.iloc[w] for w in woi]
    x1 = wol[0]
    x2 = wol[-1]
    m = (y2 - y1) / (x2 - x1)  # m is the slope at all pixels
    b = y2 - m * x2  # b is the intercept at all pixels
    y = m * np.array(wol) + b
    h = [getRvalue(spec, wvt, wvt[i], kwidth=1) - (m * wvt[i] + b) for i in woi]

    # Fit a cubic spline using wol and h
    cs_ = cs(wol, h)  # Find the minimum y value using optimization

    # ----------------------------------------------------------------
    # option 1

    # Generate a finer mesh of x values
    x_finer = np.linspace(min(wol), max(wol), 1000)  # Change 1000 to the desired number of points

    # Calculate y values corresponding to the finer mesh of x values
    y_finer = cs_(x_finer)

    # Find the index of the minimum y value
    min_y_index = np.argmin(y_finer)

    # The x value at the minimum y value
    x_at_min_y = x_finer[min_y_index]
    y_min = y_finer[min_y_index]

    # ----------------------------------------------------------------
    # option 2
    # Define a function that calculates the value of the CubicSpline at a given x
    # def spline_at_x(x_value):
    #     return cs_(x_value)

    # # Use minimize_scalar to find the x value corresponding to the minimum y
    # result = minimize_scalar(spline_at_x, bounds=(min(wol), max(wol)), method='bounded')

    # # The x value corresponding to the minimum y value
    # x_at_min_y = result.x
    # y_min = result.fun

    # calculate the full width half maximum of the band defined by wol and h
    peak_index = np.argmin(np.abs(h - y_min))

    # calculate the band area
    ba = -0.001 * getCubicSplineIntegral((wol, h))

    # calculate band area to the left of the band center
    ba_left = -0.001 * getCubicSplineIntegral((wol[:peak_index], h[:peak_index]))

    # calculate band area to the right of the band center
    ba_right = -0.001 * getCubicSplineIntegral((wol[peak_index:], h[peak_index:]))

    basym = np.log10(ba_right / ba_left)
    # basym = (ba_right-ba_left)/ba

    return basym


def getRSlope(spec, wvt, low, high, kwidth=5):
    """retrieve slope

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        kwidth (int, optional): kernel width for median filter. Defaults to 5.

    Returns:
        (array): slope image
    """
    y1 = getRvalue(spec, wvt, low, kwidth=kwidth)
    x1 = getClosestWavelength(low, wvt)
    y2 = getRvalue(spec, wvt, high, kwidth=kwidth)
    x2 = getClosestWavelength(high, wvt)
    m = (y2 - y1) / (x2 - x1)  # m is the slope
    # nmin = np.nanmin(np.where(m>-np.inf,m,np.nan))
    # s = np.where(m>-np.inf,m,nmin)
    return m


# ----------------------------------------------------------------
# read info from .sed files
# ----------------------------------------------------------------
def getReflectanceFromSed(sedFile):
    """
    Reads in a spectral energy distribution (SED) file and returns the wavelength and reflectance data.

    Args:
        sedFile (str): The path to the SED file.

    Returns:
        tuple: A tuple containing two lists - the wavelength data and the reflectance data.
    """
    with open(sedFile) as lf:
        sedInfo = np.array([line[:-1] for line in lf.readlines()])
    wvl = []
    refl = []
    i = 0
    # get index where data start
    for line in sedInfo:
        if line.__contains__("Wvl"):
            idx = i + 1
        i = i + 1
    # get data
    info = sedInfo[idx:]
    for line in info:
        b, r = line.split("\t")
        wvl.append(float(b))
        refl.append(float(r))
    return wvl, refl


def getSedFiles(sedPath):
    """
    Returns a pandas DataFrame containing reflectance values for all SED files in the given path.

    Args:
    sedPath (str): The path to the SED files.

    Returns:
    pandas.DataFrame: A DataFrame containing reflectance values for all SED files in the given path.
    """
    i = 0
    for file in glob.glob(sedPath):
        h = file.split("/")
        name = h[-1]
        wvl, r = getReflectanceFromSed(file)
        if i == 0:
            initialDict = {"Wavelength": wvl, name: r}
            df = pd.DataFrame(initialDict)
        else:
            df[name] = r
        i = i + 1
    return df


# ----------------------------------------------------------------
# read files from USGS speclib07
# ----------------------------------------------------------------
# These functions are for reading files downloaded from the USGS splib07a library of reflectance spectra
# Input is a path to the reflectance .txt file. Output is a pandas.DataFrame of the wavelength values and
# reflectance values.
def getReflectanceFromUSGS(usgsFile):
    """
    Reads reflectance data from a USGS file and returns it as a list.

    Parameters:
    usgsFile (str): The path to the USGS file.

    Returns:
    list: A list containing the reflectance values from the USGS file.
    """
    with open(usgsFile) as lf:
        fileInfo = np.array([line[:-1] for line in lf.readlines()])
    # get data
    refl = fileInfo[1:]
    refl = [float(i) for i in refl]
    for value in refl:
        if value < 0:
            i = refl.index(value)
            refl[i] = np.nan
            refl[i] = np.nanmean(refl[(i - 1) : (i + 1)])
    return refl


def getWavelengthFromUSGS(usgsFile):
    """
    Reads wavelength data from a USGS file and returns it as a list.

    Parameters:
    usgsFile (str): The path to the USGS file.

    Returns:
    list: A list containing the wavelength values from the USGS file converted to nm.
    """
    with open(glob.glob(usgsFile)[0]) as lf:
        fileInfo = np.array([line[:-1] for line in lf.readlines()])
    # get data
    wvl = fileInfo[1:]  # convert to nm
    wvl = [float(i) * 1000 for i in wvl]
    for value in wvl:
        if value < 0:
            wvl[wvl.index(value)] = np.nan
    return wvl


def getSpecFiles(usgsPath):
    """
    Reads USGS files from a directory path, retrieves the wavelength and reflectance data,
    and returns a pandas DataFrame.

    Parameters:
    usgsPath (str): The path to the directory containing the USGS files.

    Returns:
    pandas.DataFrame: A DataFrame with wavelength and reflectance data from USGS files.
    """
    i = 0
    for file in glob.glob(usgsPath):
        h = file.split("/")
        name = h[-1]
        usgsWvlPath = "/".join(h[:-1]) + "/splib07a_Wavelengths*.txt"
        print(usgsWvlPath)
        wvl = getWavelengthFromUSGS(usgsWvlPath)
        r = getReflectanceFromUSGS(file)
        if i == 0:
            initialDict = {"Wavelength": wvl, name: r}
            df = pd.DataFrame(initialDict)
        else:
            df[name] = r
        i = i + 1
    return df
