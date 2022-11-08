""" functions used for analysis scripts for 'Effects of circulation on tropical cloud feedback in high resolutions simulations', Mackie & Byrne, 2022

Anna Mackie, 2022"""

import numpy as np
from scipy import stats
import numpy.ma as ma
from scipy.optimize import curve_fit

def addr2line(x,y,ax, color = 'k', position = [0.75, 0.85], plot = True, extra_text = '', fontsize = 16):
    """ calculates and adds r^2 line to plot
    x, y: variables for correlation
    ax: axes on which to plot
    color: colour of line
    position: position of inset text containing r^2 value
    plot: True if want line plotted on plot
    extra_text: string for any additional text
    fontsize: inset text fontsize

    returns r^2 to two decimal places
    
    """
    slope, intercept, r_value, p_value, stderr = stats.linregress(x,y)
    print('slope is: ', slope, ' and standard error is:', stderr )
    if plot ==True:
        ax.plot(x, x*slope+intercept, color = color)
    txtstr = 'r$^2$ = ' + "%.2f" % r_value**2
    if p_value < 0.01:
        ax.text(position[0], position[1], extra_text + txtstr, horizontalalignment='left',
                            verticalalignment='center',transform=ax.transAxes, color = color, weight = 'bold', fontsize = fontsize)    
    else:
        ax.text(position[0], position[1], extra_text + txtstr, horizontalalignment='left',
                            verticalalignment='center',transform=ax.transAxes, color = color, fontsize = fontsize)   
    return round(r_value**2, 2)

def calc_area_mix(mnbin,A1,A2,R1,R2, deltaT):
    """ follows calculation in Byrne & Schneider, 2018. Not used"""
    dA = A2 - A1
    R1u, R1d = np.nanmean(R1[mnbin<0]), np.nanmean(R1[mnbin>0])
    dAu, dAd = np.nanmean(dA[mnbin<0]), np.nanmean(dA[mnbin>0])
    
    area = (R1u*dAu)*np.sum(mnbin[mnbin<0])/deltaT + (R1d*dAd)*np.sum(mnbin[mnbin>0])/deltaT
    mix = (np.nansum((R1 - R1u)*(dA - dAu)) + np.nansum((R1 - R1d)*(dA - dAd)))/deltaT
    return area, mix

""" the following six functions calculate two bulk metrics subsidence/ascent fraction (Fdown/Fup) and organisation index (Iorg) for both variables in pressure and z coordinates"""

def calcFdown(wa500): # m/s
    ndown = np.count_nonzero(wa500 < 0)  # vertical velocity criteria
    nup = np.count_nonzero(wa500 > 0)
    
    return ndown/(ndown + nup)

def calcFdown_hpa(wa500): # hpa/s
    ndown = np.count_nonzero(wa500 > 0)  # vertical velocity criteria
    nup = np.count_nonzero(wa500 < 0)
    
    return ndown/(ndown + nup)

def calcFup(wa500): # m/s
    ndown = np.count_nonzero(wa500 < 0)  # vertical velocity criteria
    nup = np.count_nonzero(wa500 > 0)
    
    return nup/(ndown + nup)

def calcFup_hpa(wa500): # hpa/s
    ndown = np.count_nonzero(wa500 > 0)  # vertical velocity criteria
    nup = np.count_nonzero(wa500 < 0)
    
    return nup/(ndown + nup)


def calcI(wa500):
    wdown = np.mean(wa500[np.where(wa500 < 0)])
    wup = np.mean(wa500[np.where(wa500 > 0)])
    return wup - wdown

def calcI_hpa(wa500):
    wdown = np.mean(wa500[np.where(wa500 > 0)])
    wup = np.mean(wa500[np.where(wa500 < 0)])
    return (wup - wdown)

def calcRMSE(x,y):
    """ calculates the root mean square error of two arrays"""
    return np.sqrt(np.mean((x-y)**2))

def calcStrength(wa500):
    wup = np.mean(wa500[np.where(wa500 > 0)])
    return wup

def createA(dig, b):
    """
    creates the normalised area pdf for the CRMs
    """
    import numpy as np
    num = [np.count_nonzero(dig == i) for i in range(1,b)]
    A_bin = num/np.sum(num) # normalise
    nans, xn= nan_helper(A_bin)
    A_bin[nans]= np.interp(xn(nans), xn(~nans), A_bin[~nans])
    return A_bin


def createA_GCM(dig, clat, b):
    """
    creates the normalised area pdf for GCMs weighted by latitude
    """
    import numpy as np
    A_bin = np.asarray([np.sum(clat[np.where(dig == i)]) for i in range(1,b)])
    #A_bin = num/np.sum(clat)
    nans, xn= nan_helper(A_bin)
    A_bin[nans]= np.interp(xn(nans), xn(~nans), A_bin[~nans])
    A_bin = A_bin/np.nansum(A_bin)
    return A_bin

def createR(arr, dig, b):
    """
    1. calcs the mean value of the array in each bins (taken from dig)
    2. interpolates across empty bins
    """
    arrbin = [np.nanmean(arr[dig==i]) for i in range(1,b)]
    arrbin = np.asarray(arrbin)
    nans, x= nan_helper(arrbin)
    #print(sum(nans))
    arrbin[nans]= np.interp(x(nans), x(~nans), arrbin[~nans])
    return arrbin


def createR_GCM(arr, clat, dig, b):
    """
    1. calcs the mean value of the array in each bins (taken from dig)
    2. interpolates across empty bins
    """
    arrbin = []
    for i in np.arange(1,b,1):
        if i in dig:
            arrbin = np.append(arrbin, np.sum(arr[dig==i]*clat[dig==i])/np.sum(clat[dig==i]))
        else:
            arrbin = np.append(arrbin, np.nan)
    
    
    #interp over nans
    nans, x= nan_helper(arrbin)
    #print(sum(nans))
    arrbin[nans]= np.interp(x(nans), x(~nans), arrbin[~nans])
    return arrbin

def decompose(A1,A2,R1,R2, deltaT):
    """
    input: control pdf (A1) and function (R1), and the new pdf (A2) and function (R2), as well as the temp diff (deltaT)
    outputs: total feedback (tot), thermodynamic effect (th), dynamic effect (dyn) and nonlinear effects (nl)
    """
    R1 = rpTrailingZeros(A1, R1)
    R2 = rpTrailingZeros(A2, R2)
    
    dR = R2 - R1
    dA = A2 - A1
    thermo = np.nansum(dR*A1)/deltaT
    dyn =np.nansum(dA*R1)/deltaT
    nl = np.nansum(dA*dR)/deltaT
    return thermo + dyn + nl, thermo, dyn, nl

def decompose_rangew(A1,A2,R1,R2, deltaT, bins, rn):
    """
    input: control pdf (A1) and function (R1), and the new pdf (A2) and function (R2), as well as the temp diff (deltaT).
    Also the values of the vertical velocity bins (bins) and the range to integrate over (rn)
    outputs: total feedback (tot), thermodynamic effect (th), dynamic effect (dyn) and nonlinear effects (nl) integrated
    over the specified range rn
    """
    if len(bins) != len(A1):
        print('different lengths!')
    
    st, en = np.argmin(abs(bins - rn[0])), np.argmin(abs(bins - rn[1]))
    print('integrating between', bins[st], ' and ', bins[en])
    A1, A2, R1, R2 = A1[st:en], A2[st:en], R1[st:en], R2[st:en]
    print(len(A1))
    R1 = rpTrailingZeros(A1, R1)
    R2 = rpTrailingZeros(A2, R2)
    
    dR = R2 - R1
    dA = A2 - A1
    thermo = np.nansum(dR*A1)/deltaT
    dyn =np.nansum(dA*R1)/deltaT
    nl = np.nansum(dA*dR)/deltaT
    return thermo + dyn + nl, thermo, dyn, nl

def fracChange(new, control, deltaT):
    """ calculates the fractional change between two states with a temperature change of deltaT"""
    return ((new - control)/control)*100/deltaT

def getPlevel(arr, pressure, pLev):
    """extracts the model level of a [t,k,j,i] array which most closely corresponds to pressure level pLev"""
    rt, ry, rx = np.shape(pressure)[0],np.shape(pressure)[2],np.shape(pressure)[3],
    narr = np.empty((rt, ry, rx))
    for xx in range(rx):
        for yy in range(ry):
            for tt in range(rt):
                pcol = pressure[tt, :, yy, xx]
                pCh = np.argmin(abs(pcol - pLev))
                narr[tt,yy,xx] = arr[tt, pCh, yy, xx]
    return narr

def linearise(mnbin, R):
    mask = ~np.isnan(R)
    slope, icpt, r_val, p_val, stderr = stats.linregress(mnbin[mask], R[mask])
    Rlin = slope*mnbin + icpt
    Rnl = R - Rlin
    return slope, icpt, r_val, Rlin, Rnl

def maxdA(dA):
    return np.nansum(np.abs(dA))/2


def nan_helper(y):
    """ got from: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def rpTrailingZeros(A, R):
    """
    replaces, in R, trailing zeros in A with nans
    """
    m= A!=0
    R = R.astype("float")
    R[:m.argmax()] = np.nan
    R[(m.size - m[::-1].argmax()):] = np.nan
    return R
def rpTrailingZerosA(A):
    """
    replaces, in A, trailing zeros in A with nans
    """
    m= A!=0
    A = A.astype("float")
    A[:m.argmax()] = np.nan
    A[(m.size - m[::-1].argmax()):] = np.nan
    return A

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def sortOut(arr):
    arr = np.swapaxes(arr, 1,2)
    arr = np.swapaxes(arr, 0,1)
    return arr

def vertical_average(data,plev):
    # VERTICAL_AVERAGE returns the mass-weighted average in pressure coordinates of the specified data. Note that 'plev' is in hPa. (From Mike)
    
    import numpy as np
    
    g = 9.81
    # Setup a metrix:
    vert_avg = np.zeros(np.size(data,1))

    # Loop over lat/lon:
    for lat_index in np.arange(0, np.size(data,1)):

        # Remove any NaNs:
        data_vec = np.squeeze(data[:,lat_index])
        plev_vec = 100 * plev
        plev_vec = plev_vec[np.logical_not(np.isnan(data_vec))]
        data_vec = data_vec[np.logical_not(np.isnan(data_vec))]

        # Integrate using the trapezoid rule:
        vert_avg[lat_index] = (np.trapz(data_vec[::-1],plev_vec[::-1]) / g) / (np.trapz(np.ones((1,np.size(data_vec[::-1]))),plev_vec[::-1]) / g)
        
    return vert_avg

######## I org code, taken from RCEMIP webpage http://myweb.fsu.edu/awing/files/IORG.py
def ecdf(x_inputs):
    """
    Calculates the empirical cumulative distribution function of x_inputs
    Arguments
        x_inputs: 1D array, that the ECDF is calculated from
    Returns
        y_values: 1D array, the ECDF
        x_values: 1D array, the values ECDF is evaluated at using x_inputs
    """

    import numpy as np

    x_sorted = np.sort(x_inputs)
    x_values = np.linspace(start=min(x_sorted),
                           stop=max(x_sorted),
                           num=len(x_sorted))

    y_values = np.zeros(np.shape(x_values))
    for i in range(len(x_values)):
        temp  = x_inputs[x_inputs <= x_values[i]]
        value = np.float64(len(temp))/np.float64(len(x_inputs))
        y_values[i] = value

    return(x_values,y_values)

def iorg_crm(w,wthreshold=0):
    """
    Calculates the Organization Index for a cartesian domain
    Arguments
        w: 2-D array of demensions (y,x) of OLR or 500hPa vertical
            acceleration RCEMIP (Wing et al., 2018) uses OLR
        wthreshold: integer, 0 for w being an array of OLR, 1 for
            vertical acceleration
    Returns
        iorg:
    """

    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import measurements
    from scipy.ndimage import label
    import numpy as np

    # create a masked array of convective entities
    if wthreshold != 1:
        if wthreshold != 0:
            print('incorrect threshold flag, using default (OLR<173)')
        wmask = (w<173)*1
    if wthreshold == 1:
        wmask = (w>0.5)*1

    # duplicates domain in all directions to account for convection on borders
    wmaskbig = np.concatenate([wmask,wmask,wmask],axis=0)
    wmaskbig = np.concatenate([wmaskbig,wmaskbig,wmaskbig],axis=1)

    # four point convective connection
    sss = [[0,1,0],
           [1,1,1],
           [0,1,0]]

    # finds connected convective entities, the number of clusters,
      # and the centroid of each cluster
    Conn = label(wmaskbig,structure=sss)[0]
    nentities = label(wmaskbig)[1]
    centroids = measurements.center_of_mass(wmaskbig,Conn,range(1,nentities+1))

    nnd,IDX,num = [],[],0
    if nentities > 1:
        # finds the nearest neighbor of each convective cluster
        classifier = NearestNeighbors(n_neighbors=1)
        for i in range(0,nentities):
            if centroids[i][0] >= np.shape(w)[0] \
               and centroids[i][0] <= (np.shape(w)[0]*2)-1 \
               and centroids[i][1] >= np.shape(w)[1] \
               and centroids[i][1] <= (np.shape(w)[1]*2)-1:
                num += 1
                classifier.fit(np.array(centroids[0:i]+centroids[i+1:]))
                m,n = classifier.kneighbors(np.reshape(centroids[i],(1,2)))
                IDX.append(n)
                nnd.append(m)

        if len(nnd) > 1:
            IDX = np.squeeze(np.array(IDX))
            nnd = np.squeeze(np.array(nnd))
        if len(nnd) <= 1:
            IDX = np.array(IDX)
            nnd = np.array(nnd)

        if len(nnd) > 0:
            # calculates ECDF of the nearest neighbors idstances
            x_values_ecdf,y_values_ecdf = ecdf(nnd)

            # calculates the poisson distribution of w
            lam = num/(np.shape(w)[0]*np.shape(w)[1])
            dd = np.array(x_values_ecdf)
            poisson = 1-np.exp(-1*lam*np.pi*dd**2)
            cdf_theory = poisson

            # calculates the area under the plot of the poisson vs ECDF
            iorg = np.trapz(y_values_ecdf,poisson)

        else:
            cdf_nnd = 0
            d = 0
            cdf_theory = 0
            iorg = 0
            y_values_ecdf,x_values_ecdf,poisson,iorg = 0,0,0,0

    else:
        cdf_nnd = 0
        d = 0
        cdf_theory = 0
        iorg = 0
        y_values_ecdf,x_values_ecdf,poisson,iorg = 0,0,0,0

    return(iorg)

def iorg_gcm(w,wthreshold=0):
    """
    Calculates the Organization Index for a speherical domain
    Arguments
        w: 2-D array of demensions (y,x) of OLR or 500hPa vertical
            acceleration RCEMIP (Wing et al., 2018) uses OLR
        wthreshold: integer, 0 for w being an array of OLR, 1 for
            vertical acceleration
    Returns
        iorg:
    """

    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import measurements
    from scipy.ndimage import label
    import numpy as np

    # create a masked array of convective entities
    if wthreshold != 1:
        if wthreshold != 0:
            print('incorrect threshold flag, using default (OLR<173)')
        wmask = (w<173)*1
    if wthreshold == 1:
        wmask = (w>0.5)*1

    # duplicates domain in all directions to account for convection on borders
    wmaskbig = np.concatenate([wmask,wmask,wmask],axis=1)

    # four point convective connection
    sss = [[0,1,0],
           [1,1,1],
           [0,1,0]]

    # finds connected convective entities, the number of clusters,
      # and the centroid of each cluster
    Conn = label(wmaskbig,structure=sss)[0]
    nentities = label(wmaskbig)[1]
    centroids = measurements.center_of_mass(wmaskbig,Conn,range(1,nentities+1))

    nnd,IDX,num = [],[],0
    if nentities > 1:
        # finds the nearest neighbor of each convective cluster
        classifier = NearestNeighbors(n_neighbors=1)
        for i in range(0,nentities):
            if centroids[i][1] >= np.shape(w)[1] \
               and centroids[i][1] <= (np.shape(w)[1]*2)-1:
                num += 1
                classifier.fit(np.array(centroids[0:i]+centroids[i+1:]))
                m,n = classifier.kneighbors(np.reshape(centroids[i],(1,2)))
                IDX.append(n)
                nnd.append(m)

        if len(nnd) > 1:
            IDX = np.squeeze(np.array(IDX))
            nnd = np.squeeze(np.array(nnd))
        if len(nnd) <= 1:
            IDX = np.array(IDX)
            nnd = np.array(nnd)

        if len(nnd) > 0:
            # calculates ECDF of the nearest neighbors idstances
            x_values_ecdf,y_values_ecdf = ecdf(nnd)

            # calculates the poisson distribution of w
            lam = num/(np.shape(w)[0]*np.shape(w)[1])
            dd = np.array(x_values_ecdf)
            poisson = 1-np.exp(-1*lam*np.pi*dd**2)
            cdf_theory = poisson

            # calculates the area under the plot of the poisson vs ECDF
            iorg = np.trapz(y_values_ecdf,poisson)

        else:
            cdf_nnd = 0
            d = 0
            cdf_theory = 0
            iorg = 0
            y_values_ecdf,x_values_ecdf,poisson,iorg = 0,0,0,0
    else:
        cdf_nnd = 0
        d = 0
        cdf_theory = 0
        iorg = 0
        y_values_ecdf,x_values_ecdf,poisson,iorg = 0,0,0,0

    return(iorg)

def calc_iorg(w_f,geometry,threshold=0,tbeg=-999,tend=-999):
    """
    Loops through a time range to calculate Iorg
    Arguments
        w_f: 3-D array (time, y, x) of the convection data
        geometry: string to determine which IORG calculation to use,
            acceptable input is 'cartesian' or 'spherical'
        threshold: optional integer (default=0) determining which
            type of convection data is being used, 0 for olr or 1 for
            vertical velocity
        tbeg: optional integer (default=-999) determining which time
            integer to start at, -999 uses 25 days from the end
        tend: optional integer (default=-999) determining which time
            integer to end at, -999 uses the end of the data
    Returns
        iorg_f: 1-D array of Iorg data from time index tbeg to tend
    """

    import numpy as np

    if geometry != 'cartesian' and geometry != 'spherical':
        print('inapproriate geometry entry, choices are ``cartesian`` or ``spherical``')
        return(np.nan)
    else:
        if tend == -999:
            tend = np.shape(w_f)[0]
        if tbeg == -999:
            tbeg = 1800

                # loop through time range
        cdf_nnd_f,d_f,cdf_theory_f,iorg_f = [],[],[],[]
        for v in range(tbeg,tend):
            if v%24 == 0:
                print('   processing day %d of %i\r'%(v/24,int(tend/24)), end='')
            # calculate Iorg
            if geometry == 'cartesian':
                result = iorg_crm(w_f[v,:,:],threshold)
            if geometry == 'spherical':
                result = iorg_gcm(w_f[v,:,:],threshold)
            iorg_f.append(result)
        
        iorg_f = np.array(iorg_f)

        return(iorg_f)