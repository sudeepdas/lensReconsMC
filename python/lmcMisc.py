from flipper import *
import scipy.interpolate

def makeTemplate(l,Fl,ftMap):
    """
    Given 1d function Fl of l, creates the 2d version
    of Fl on 2d k-space defined by ftMap
    """
    tck = scipy.interpolate.splrep(l,Fl)
    template = scipy.interpolate.splev(ftMap.modLMap.ravel(),tck)
    template = numpy.reshape(template,[ftMap.Ny,ftMap.Nx])
    return template

def trimShiftKMap(kMap,elTrim,Lx,Ly,lx,ly):
    """
    @brief Trims a 2-D powerMap at a certain Lx, Ly, and returns the trimmed power2D object. Note
    that the pixel scales are adjusted so that the trimmed dimensions correspond
    to the same sized map in real-space (i.e. trimming -> poorer resolution real space map).
    Can be used to shift maps.
    @pararm elTrim real >0 ; the l to trim at
    @return power2D instance
    """
    
    assert(elTrim>0.)
    idx = numpy.where((lx < elTrim+Lx) & (lx > -elTrim+Lx))
    idy = numpy.where((ly < elTrim+Ly) & (ly > -elTrim+Ly))
    
    trimA = kMap[idy[0],:] # note this used to be kM!!! not kMap
    trimB = trimA[:,idx[0]]
    
    return trimB

def mapFlip(kMap,lx,ly):
    kMapNew = kMap.copy()
    for i in ly:
        for j in lx:
            iy0 = numpy.where(ly == i)
            ix0 = numpy.where(lx == j)
            iy = numpy.where(ly == -i)
            ix = numpy.where(lx == -j)
            
            try:
                kMapNew[iy0[0],ix0[0]] = kMap[iy[0],ix[0]]
            except:
                pass
    return kMapNew
        
        
