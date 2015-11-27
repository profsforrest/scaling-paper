import numpy as np
import scipy.stats as stats
from collections import namedtuple

RMAResult = namedtuple('RMAResult',['slope','intercept','ci','slope_ci1','intercept_ci1','slope_ci2','intercept_ci2','RSquare'])

def RMARegression(X, Y, alpha=0.05):
    """
    GMREGRESS Geometric Mean Regression (Reduced Major Axis Regression).
    Model II regression should be used when the two variables in the
    regression equation are random and subject to error, i.e. not
    controlled by the researcher. Model I regression using ordinary least
    squares underestimates the slope of the linear relationship between the
    variables when they both contain error. According to Sokal and Rohlf
    (1995), the subject of Model II regression is one on which research and
    controversy are continuing and definitive recommendations are difficult
    to make.
    GMREGRESS is a Model II procedure. It standardize variables before the
    slope is computed. Each of the two variables is transformed to have a
    mean of zero and a standard deviation of one. The resulting slope is
    the geometric mean of the linear regression coefficient of Y on X.
    Ricker (1973) coined this term and gives an extensive review of Model
    II regression. It is also known as Standard Major Axis.
    b, bintr, bintjm = gmregress(X,Y,alpha)
    returns the vector B of regression coefficients in the linear Model II and
    a matrix BINT of the given confidence intervals for B by the Ricker (1973)
    and Jolicoeur and Mosimann (1968)-McArdle (1988) procedure.
    gmregress treats NaNs in X or Y as missing values, and removes them.
    Syntax: function b, bintr, bintjm = gmregress(X, Y, alpha)
    Example. From the Box 14.12 (California fish cabezon [Scorpaenichthys
    marmoratus]) of Sokal and Rohlf (1995). The data are:
    x = [14, 17, 24, 25, 27, 33, 34, 37, 40, 41, 42]
    y = [61, 37, 65, 69, 54, 93, 87, 89, 100, 90, 97]
    Calling on Matlab the function:
    b, bintr, bintjm = gmregress(x,y)
    Answer is:
    b = 12.1938    2.1194
    bintr = -10.6445   35.0320
            1.3672    2.8715
    bintjm = -14.5769   31.0996
            1.4967    3.0010
    http://www.mathworks.com/matlabcentral/fileexchange/27918-gmregress
    References:
        Jolicoeur, P. and Mosimann, J. E. (1968), Intervalles de confiance pour
            la pente de l'axe majeur d'une distribution normale
            bidimensionnelle. Biometrie-Praximetrie, 9:121-140.
        McArdle, B. (1988), The structural relationship: regression in biology.
            Can. Jour. Zool. 66:2329-2339.
        Ricker, W. E. (1973), Linear regression in fishery research. J. Fish.
            Res. Board Can., 30:409-434.
        Sokal, R. R. and Rohlf, F. J. (1995), Biometry. The principles and
            practice of the statistics in biologicalreserach. 3rd. ed.
            New-York:W.H.,Freeman. [Sections 14.13 and 15.7]
    """
    X, Y = list(map(np.asanyarray, (X, Y)))

    n = len(Y)
    S = np.cov(X, Y)
    SCX = S[0, 0] * (n - 1)
    SCY = S[1, 1] * (n - 1)
    SCP = S[0, 1] * (n - 1)
    
    R = np.corrcoef(X, Y)
    r = R[0, 1]
    
    v = np.sign(r)*np.sqrt(SCY / SCX)  # Slope.
    u = Y.mean() - X.mean() * v  # Intercept.
    b = np.r_[u, v]

    SCv = SCY - (SCP ** 2) / SCX
    N = SCv / (n - 2)
    sv = np.sqrt(N / SCX)
    t = stats.t.isf(alpha / 2, n - 2)

    vi = v - t * sv  # Confidence lower limit of slope.
    vs = v + t * sv  # Confidence upper limit of slope.
    ui = Y.mean() - X.mean() * vs  # Confidence lower limit of intercept.
    us = Y.mean() - X.mean() * vi  # Confidence upper limit of intercept.
    bintr = np.r_[np.c_[ui, us], np.c_[vi, vs]]

    R = np.corrcoef(X, Y)
    r = R[0, 1]

    F = stats.f.isf(alpha, 1, n - 2)
    B = F * (1 - r ** 2) / (n - 2)

    a = np.sqrt(B + 1)
    c = np.sqrt(B)
    qi = v * (a - c)  # Confidence lower limit of slope.
    qs = v * (a + c)  # Confidence upper limit of slope.
    pi = Y.mean() - X.mean() * qs  # Confidence lower limit of intercept.
    ps = Y.mean() - X.mean() * qi  # Confidence upper limit of intercept.
    bintjm = np.r_[np.c_[pi, ps], np.c_[qi, qs]]
    

    return RMAResult(slope=b[1],intercept=b[0],ci=1-alpha,intercept_ci1 = bintr[0],slope_ci1 = bintr[1],intercept_ci2=bintjm[0],slope_ci2=bintjm[1],RSquare=r**2)