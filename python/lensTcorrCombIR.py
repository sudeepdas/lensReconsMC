from flipper import *
#from scipy.integrate import 
from scipy.interpolate import splev,splrep
from flipper import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2
from scipy import interpolate
from scipy import *
import os
import random
import sys
import pickle
import time

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
    #print 'where x:', lx[idx]
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
          #  print 'iyix', iy, ix, iy0, ix0
            try:
           #     print 'yaay'
                kMapNew[iy0[0],ix0[0]] = kMap[iy[0],ix[0]]
            #    print kMapNew[iy0[0],ix0[0]], kMap[iy0[0],ix0[0]]
            except:
            #    print 'noooo'
                pass
    return kMapNew


class lensFilters:
    """
    @brief Class describing lensing filters
    """
    def __init__(self):
        """
        @brief Initializes the lensFilters
        """
        self.el = None
        self.cl = None
        self.lowPassElSpaceRep = None
        self.highPassElSpaceRep = None
        self.normalization = None
        self.lowPassFilter2D = None
        self.highPassFilter2D = None
        self.lowPassKMap = None
        self.highPassKMap = None
        self.kappaMap = None
        
        
    
    def normalize(self,accuracyBoost=1):
        """
        @brief Computes the normalization  N(L) given the  lensFilters.
        <i> Superceded by Normalize2D </i>
        @param accuracyBoost Increases accurcy of internal interpolations.
        
        """
        L = self.el
        
        # Make a array, Larray, at which N(L) will be computed and later
        # interpolated onto all integers.
        
        nLinear = 10         # Always sample the first nLinear indices
        Llinear = L[0:nLinear]
        nPoints = 60*accuracyBoost
        Larray = (Llinear.max()+1)\
                 *(L.max()/(Llinear.max()+1))**(numpy.arange(nPoints)/(nPoints-1.0))
        Larray = numpy.append(Llinear,Larray)
        
        self.normalization = L.copy()
        norm = Larray.copy()
        
        for i in xrange(len(Larray)):
            # Call NofL to compute normalization
            norm[i] = self.NofL(Larray[i],accuracyBoost=accuracyBoost)
            #trace.issue("lensRecons.lensTools",5,"norm = %f"%norm[i])
            
        # Spline in log space - this is more stable
        logSplNorm = splrep(Larray,numpy.log(norm),k=3)
        self.normalization = numpy.exp(splev(L,logSplNorm))
        
        
    def NofL(self,L,accuracyBoost=1):
        """
        @brief Computes the normalization at each L
        <i> Superceded by NofLTwoD </i>
        @param L the L-value at which NofL is being computed
        
        """

        nOfL = 0.
        #trace.issue("lensRecons.NofL",5, "L = %f"%L)

        # Do the theta integral first
        # by sampling the thetas at nPnts points
        
        nPnts = 1000*accuracyBoost
        thetas = numpy.arange(nPnts)/(nPnts-1.0)*numpy.pi*2.0
        dtheta  = thetas[1]-thetas[0]
        
        nelinear = 10
        ellinear = self.el[0:nelinear]
        nPoints = 180*accuracyBoost
        elarray = (ellinear.max()+1)\
                 *(self.el.max()/(ellinear.max()+1))**(numpy.arange(nPoints)/(nPoints-1.0))
        elarray = numpy.append(ellinear,elarray)
        
        nOfLarray = elarray.copy()*0.
        
        #for i in xrange(len(self.el)/3):
            #(integ,err) = quadrature(normKernel,0.,2.*numpy.pi,args=[L,el,cl],tol=1e-5)
            
        #    integ = (normKernel(thetas,L,self.el[i],self)).sum()
            
        #    nOfL +=  integ*dtheta

        for i in xrange(len(elarray)):
            integ = (normKernel(thetas,L,elarray[i],self)).sum() 
            nOfLarray[i] = integ*dtheta

        spl = splrep(elarray,nOfLarray,k=3)
        nOfL = splint(self.el.min(),self.el.max(),spl)
        nOfL *= 2.0/L**2
        nOfL = 1./nOfL
        return nOfL

    def normKernel(theta,L,el,lensFilters):
        """
        @brief Normalization kernel used by NofL
        """
        
        #Here elPrime = \vec L-\vec el 
        LdotEl = L*el*numpy.cos(theta)
        elPrime = numpy.sqrt(L**2+el**2-2*LdotEl)
        LdotElPrime = L**2-LdotEl
        cEl = numpy.exp(splev(el,lensFilters.logSplCl))
        cElPrime = numpy.exp(splev(elPrime,lensFilters.logSplCl))
        F_W_El = splev(el,lensFilters.splineHP)
        F_G_ElPrime = splev(elPrime,lensFilters.splineLP)
        nk = el/(2*numpy.pi)**2*F_W_El*LdotElPrime*F_G_ElPrime*(LdotElPrime*cElPrime+LdotEl*cEl)
        
        idx = numpy.where((elPrime < lensFilters.el.min()) | (elPrime > lensFilters.el.max()))
        nk[idx] = 0.
        return nk 


    def makeKappaMap(self,trimAtL=None):
        """
        @brief Finds the normalization and noise bias of the lensing filter.
        This is different from normalize() as it does all operations in
        the 2-D L-space without resorting to interpolations.
        Uses NofL2D.
        @param trim2DAtL trim the 2D calculation at a given L
        
        """
        
        ft = self.powerMap1
        if trimAtL != None:
            ft = ft.trimAtL(trimAtL)
        #print ftLow.kMap, 'km'
        lx = numpy.fft.fftshift(ft.lx)
        ly = numpy.fft.fftshift(ft.ly)
        self.deltaLy = numpy.abs(ly[0]-ly[1])
        self.deltaLx = numpy.abs(lx[0]-lx[1])
        modLMap = numpy.fft.fftshift(ft.modLMap)
        thetaMap = numpy.fft.fftshift(ft.thetaMap)
        cosTheta = numpy.cos(thetaMap*numpy.pi/180.)
        self.modLMap = modLMap
        self.LhatDotEl = modLMap*cosTheta
        self.dims = [ft.Ny,ft.Nx]
        
        
        ftb = self.powerMap1
        lxb = numpy.fft.fftshift(ftb.lx)
        lyb = numpy.fft.fftshift(ftb.ly)
        ftKappa = self.map1
        kappaMap = trimShiftKMap(numpy.fft.fftshift(ftKappa),trimAtL,0,0,lxb,lyb)
        highPassFilteredMap = trimShiftKMap(numpy.fft.fftshift(self.highPassKMap),trimAtL,0,0,lxb,lyb)
        hpFilterM = trimShiftKMap(numpy.fft.fftshift(self.highPassF0),trimAtL,0,0,lxb,lyb)
        #newLpFilterM = trimShiftKMap(numpy.fft.fftshift(self.lowPassF0),trimAtL,0,0,lxb,lyb)
        lowPassBackup0 = self.lowPassKMap.copy()
        lowPassBackup = numpy.fft.fftshift(lowPassBackup0)
        lpFilterB0 = self.lowPassF0.copy()
        lpFilterB = numpy.fft.fftshift(lpFilterB0)
        lowPassFilteredMap0 = self.lowPassKMap.copy()
        lowPassFilteredMap = numpy.fft.fftshift(lowPassFilteredMap0)
        lpFilterM0 = self.lowPassF0.copy()
        lpFilterM = numpy.fft.fftshift(lpFilterM0)
        print lxb,'this is length'
        lowPassFilteredMap = mapFlip(lowPassBackup,lxb,lyb)#numpy.conjugate(lowPassBackup)
        lpFilterM = mapFlip(lpFilterB,lxb,lyb)
        #for i in xrange(len(lyb)):
         #   for j in xrange(len(lxb)):
          #      try:
           #         lowPassFilteredMap[i,j] = lowPassBackup[len(lyb)-1-i,len(lxb)-1-j]
            #    except:
             #       pass
        lxMap = highPassFilteredMap.copy()
        lyMap = highPassFilteredMap.copy()
        for p in xrange(len(ly)):
            lxMap[p,:] = lx[:]

        for q in xrange(len(lx)):
            lyMap[:,q] = ly[:]  
        
        count = 0.
        normalizationNew = kappaMap.copy()
        noisebias = kappaMap.copy()
        cl0 = trimShiftKMap(numpy.fft.fftshift(self.crossP1),trimAtL,0,0,lxb,lyb)
        clN = trimShiftKMap(numpy.fft.fftshift(self.autoPM),trimAtL,0,0,lxb,lyb)
         
        newLpFilterM = trimShiftKMap(numpy.fft.fftshift(self.lowPassF0),trimAtL,0,0,lxb,lyb)
        newHpFilterMFlip = mapFlip(numpy.fft.fftshift(self.highPassF0),lxb,lyb)
        clNFlip = mapFlip(numpy.fft.fftshift(self.autoPM),lxb,lyb)
        
        
        for i in xrange(len(ly)):
            a = time.time()
            for j in xrange(len(lx)):
                count += 1
                kappaMap[i,j], normalizationNew[i,j], noisebias[i,j], ele, timeratio, timeratio2 = self.kappaIntegral(lx[j],\
ly[i],lxMap,lyMap,lxb,lyb,highPassFilteredMap,lowPassFilteredMap,hpFilterM,lpFilterM,trimAtL,cl0,clN,clNFlip,newLpFilterM,newHpFilterMFlip)##all fftshifted
                if not (normalizationNew[i,j] ==  normalizationNew[i,j]):
                    normalizationNew[i,j] = normalizationNew[i,j-1]
                if not (noisebias[i,j] ==  noisebias[i,j]):
                    noisebias[i,j] = noisebias[i,j-1]
                if j == 50+i:
                    print 'values k,n1,n2,el', kappaMap[i,j], normalizationNew[i,j]
                    print 'time ratio:', timeratio, timeratio2
            b = time.time()
            print i, len(ly), 'time=', (b-a)
        #########
        ftbla = ft.copy()
        ftbla.powerMap = numpy.real(numpy.fft.ifftshift(normalizationNew))
        ftbla.plot(zoomUptoL =3000, log=True)
        pylab.savefig('2Dnorm.png')
        lL,lU,lBin,clNo1,err,w = ftbla.binInAnnuli(os.environ['LENSRECONSMC_DIR']+os.path.sep+'params/BIN_100_LOG')  
        ###########
        self.kappaMap = numpy.fft.ifftshift(kappaMap)
        #ft.kMap = numpy.fft.ifftshift(kappaMap)
        ftransf = ft.copy()
        
        return ftransf, numpy.fft.ifftshift(kappaMap), normalizationNew, noisebias



    def kappaIntegral(self,Lx,Ly,lx,ly,lxb,lyb,hpKMap,lpKMapMinl,hpf,lpf,trimAtL,cl0,clN,clNFlip,lpFTrim,hpFFlip):
        """
        @brief Computes kappa at a given L
        @param L the multipole to calculate at.
        """
        ####### all input quantities should be fftShifted
        one = time.time()
        Lsqrd = Lx**2. + Ly**2.
        LdotEl = Lx*lx + Ly*ly
        LdotElPrime = Lx**2. + Ly**2. - (Lx*lx+Ly*ly)
        three = time.time()
        lowPassShifted = trimShiftKMap(lpKMapMinl,trimAtL,-Lx,-Ly,lxb,lyb)
        four = time.time()

        highPass = hpKMap
        kint = 1./(2.*numpy.pi)**2.*highPass*lowPassShifted*LdotElPrime*\
             self.deltaLx*self.deltaLy
        two = time.time()
        diff = two - one
        
        kappa = kint.sum()
        
        diff2 = four - three

        clL = trimShiftKMap(numpy.fft.fftshift(self.crossP1),trimAtL,-Lx,-Ly,lxb,lyb)
        lpsf = trimShiftKMap(lpf,trimAtL,-Lx,-Ly,lxb,lyb)#(hpFFlip,trimAtL,-Lx,-Ly,lxb,lyb)
        nint = 2./Lsqrd*1./(2.*numpy.pi)**2.*(hpf*lpsf)*LdotElPrime*(LdotElPrime*clL+LdotEl*cl0)*\
             self.deltaLx*self.deltaLy
        norm = 1./nint.sum()
        kappa *= norm

        clNL = trimShiftKMap(clNFlip,trimAtL,-Lx,-Ly,lxb,lyb)
        hpFFlipTrim = trimShiftKMap(hpFFlip,trimAtL,-Lx,-Ly,lxb,lyb)

        firstTermNB = (hpf*lpsf)*numpy.conjugate(hpf*lpsf)*LdotElPrime**2.*clL*cl0#clN*clNL
        secondTermNB = numpy.conjugate(hpf*lpsf)*(lpFTrim*hpFFlipTrim)*LdotElPrime*LdotEl*clL*cl0#*clN*clNL
        noiseBiasInt = (numpy.abs(norm))**2.*1./(2.*numpy.pi)**2.*(firstTermNB+0.*secondTermNB)*\
            self.deltaLx*self.deltaLy
        noiseBias = noiseBiasInt.sum()

        return kappa, norm, noiseBias, numpy.sqrt(Lsqrd), diff, diff2


    def normalizeTwoD(self,trimAtL=None):
        """
        @brief Finds the normalization and noise bias of the lensing filter.
        This is different from normalize() as it does all operations in
        the 2-D L-space without resorting to interpolations.
        Uses NofL2D.
        @param trim2DAtL trim the 2D calculation at a given L
        
        """
        ########### we are now using all mean cross cls for EVERY cl in the normalization and noise bias. technically the normalization should be with the unlensed
        ########### cls. All quantities are in 2D now.
        ############# in first paragraph: trimming, fftshifting the maps and setting up the self.variables
        ft = fftTools.powerFromLiteMap(self.map1)
        ftbback = fftTools.powerFromLiteMap(self.map1)
        ftLow = ft.copy()
        ftCl = ft.copy()
        ftCl.powerMap = self.crossP1.copy()
        ftLow.powerMap = self.lowPassFilter2D.copy()
        ftHigh = ft.copy()
        ftHigh.powerMap = self.highPassFilter2D.copy()
        elmax = self.el.max()
        if trimAtL != None:
            ft = ft.trimAtL(trimAtL)
            ftCl = ftCl.trimAtL(trimAtL)
            ftHighOrig = ftHigh.trimAtL(trimAtL)
            ftLowOrig = ftLow.trimAtL(trimAtL)
            ftHigh = ftHigh.trimAtL(trimAtL)
            ftLow = ftLow.trimAtL(trimAtL)
            elmax = trimAtL
        #print ftLow.kMap, 'km'
        self.lyTrim = numpy.fft.fftshift(ftLow.ly)
        self.lxTrim = numpy.fft.fftshift(ftLow.lx)
        self.lowPassFilter2DBig = self.lowPassFilter2D.copy()
        self.highPassFilter2DBig = self.highPassFilter2D.copy()
        self.lowPassFilter2D = numpy.fft.fftshift(ftLowOrig.powerMap)
        self.highPassFilter2D = numpy.fft.fftshift(ftHighOrig.powerMap)
        ####### making one big flipped and shifted map
        self.lowPassBigTrimFftshift = numpy.fft.fftshift(mapFlip(self.lowPassFilter2DBig,ftbback.lx,ftbback.ly))
        self.highPassBigTrimFftshift = numpy.fft.fftshift(mapFlip(self.highPassFilter2DBig,ftbback.lx,ftbback.ly))
        self.crossPowerBFF = numpy.fft.fftshift(mapFlip(self.crossP1,ftbback.lx,ftbback.ly))
        self.crossPowerTF = numpy.fft.fftshift(ftCl.powerMap)
        self.lxBigShift = numpy.fft.fftshift(ftbback.lx)
        self.lyBigShift = numpy.fft.fftshift(ftbback.ly)

        ###### defining some more useful variables
        lx = numpy.fft.fftshift(ft.lx)
        ly = numpy.fft.fftshift(ft.ly)
        self.deltaLy = numpy.abs(ly[0]-ly[1])
        self.deltaLx = numpy.abs(lx[0]-lx[1])
        
        modLMap = numpy.fft.fftshift(ft.modLMap)
        thetaMap = numpy.fft.fftshift(ft.thetaMap)
        cosTheta = numpy.cos(thetaMap*numpy.pi/180.)
        self.modLMap = modLMap
        self.LhatDotEl = modLMap*cosTheta
        ##########array of Ls for norm/nb to be evaluated at
        el = numpy.ravel(modLMap)
        cEl = numpy.exp(splev(el,self.logSplCl))
        self.cEl = numpy.reshape(cEl,[ft.Ny,ft.Nx])
        
        self.dims = [ft.Ny,ft.Nx]

        shiftedlx = self.lxTrim
        zeroloc = numpy.where(self.lxTrim == 0.)
        elArray = self.lxTrim[zeroloc[0]:]
        
        print elArray
        norm = elArray.copy()*0.
        noiseBias = elArray.copy()*0.
        
        for i in xrange(len(elArray)):
            norm[i], noiseBias[i] = self.NofLTwoD(elArray[i],trimAtL)
            print i, norm[i], noiseBias[i]
            if numpy.isinf(norm[i]):
                norm[i] = norm[i-1]
            if not (norm[i] == norm[i]):
                norm[i] = norm[i-1]
           # if (norm[i]<0.):
            #    norm[i] = norm[i-1]
            if numpy.isinf(noiseBias[i]):
                noiseBias[i] = noiseBias[i-1]
            if (noiseBias[i]<0.):
                noiseBias[i] = noiseBias[i-1]
            if numpy.isinf(1./noiseBias[i]):
                noiseBias[i] = noiseBias[i-1]
            if not (noiseBias[i] == noiseBias[i]):
                noiseBias[i] = noiseBias[i-1]
            print i, norm[i], noiseBias[i]
        
        for i in xrange(len(norm)):
            print i, elArray[i], norm[i], noiseBias[i]
        Spl = splrep(elArray,norm,k=2)############changed degree from k=3
        SplNB = splrep(elArray,noiseBias,k=2)        
        
        self.normalization = splev(self.el,Spl)
        self.noiseBias = splev(self.el,SplNB)#numpy.exp(splev(self.el,logSplNB))#splev(self.el,SplNB)#
        for j in xrange(len(self.noiseBias)):
            if numpy.isinf(self.noiseBias[j]):
                self.noiseBias[j] = self.noiseBias[j-1]
        for j in xrange(len(self.normalization)):
            if numpy.isinf(self.normalization[j]):
                self.normalization[j] = self.normalization[j-1]


        
    def NofLTwoD(self,L,trimAtL):
        """
        @brief Computes normalization and noisebias at a given L
        @param L the multipole to calculate at.
        """
        L = L*1.0
        LdotEl = L*self.LhatDotEl
        elPrime = numpy.sqrt(L**2+self.modLMap**2-2.0*LdotEl)

        LdotElPrime = L**2 - LdotEl
        
        F_G_El = self.lowPassFilter2D.copy()
        F_W_El = self.highPassFilter2D.copy()

        clCrossPrime = trimShiftKMap(self.crossPowerBFF,trimAtL,-L,0.,self.lxBigShift,self.lyBigShift)
        F_G_ElPrime = trimShiftKMap(self.lowPassBigTrimFftshift,trimAtL,-L,0.,self.lxBigShift,self.lyBigShift)
        F_W_ElPrime = trimShiftKMap(self.highPassBigTrimFftshift,trimAtL,-L,0.,self.lxBigShift,self.lyBigShift)

        nk = 2./(2*numpy.pi)**2*LdotElPrime*(LdotElPrime*clCrossPrime + LdotEl*self.crossPowerTF)*\
             self.deltaLx*self.deltaLy/L**2*(F_W_El*F_G_ElPrime)##clCrossPrime####### note here we use the measured 2d mean cross power for the unlensed spec - not quite correct

        idx = numpy.where(elPrime > 8192.)
        nk[idx] = 0.
        
        NL = 1./nk.sum() #### normalization

        
        nb = NL**2/(2.0*numpy.pi)**2*(self.crossPowerTF*clCrossPrime*F_W_El**2*F_G_ElPrime**2*LdotElPrime**2\
                                 + self.crossPowerTF*clCrossPrime*F_W_El*F_W_ElPrime*F_G_El*F_G_ElPrime*LdotElPrime*LdotEl)*\
                                 self.deltaLx*self.deltaLy ###### use 2d cross power maps now - more correct

        nb[idx] = 0.
        NBiasL = nb.sum() ###### noise bias
        return NL, NBiasL
    
    
class  lensFiltersRealSpace(lensFilters):
    """
    @brief Class to construct lens filters through real space operations
    
    """
    def __init__(self,map1,map2,\
                 el,clUnlensed,\
                 elObs,clAutoObs1,clAutoObs2,\
                 clCrossObs,fwhm):
        
        self.el = el
        self.cl = clUnlensed
        self.logSplCl = splrep(el,numpy.log(clUnlensed),k=3)
        
        self.elObs  = elObs
        self.clAutoObs1 = clAutoObs1
        self.logSplCl_11 = splrep(elObs,numpy.log(clAutoObs1),k=3)
        self.clAutoObs2 = clAutoObs2
        self.logSplCl_22 = splrep(elObs,numpy.log(clAutoObs2),k=3)
        self.clCrossObs = clCrossObs
        self.logSplCl_12 = splrep(elObs,numpy.log(clCrossObs),k=3)
        self.map1 = map1.copy()
        self.map2 = map2.copy()
        self.fwhm = fwhm
        
    def applyLowPass(self, fwhm = 1.4):
        """
        @brief Real space convolution mimicking a low pass filter to be applied to
        a liteMap.  
        Currently a Gaussian Smoothing.
        
        @param FWHM FWHM of Gaussian in arcmin
        @retun A low pass filtered map.
        """
        
        lowPassMap = self.map2.convolveWithGaussian(fwhm,nSigma=5.0)
        
        fwhmInRad = fwhm*numpy.pi/(180.*60.)
        self.lowPassElSpaceRep = numpy.exp(-self.el**2*fwhmInRad**2/(16.0*numpy.log(2.)))
        self.splineLP = splrep(self.el,self.lowPassElSpaceRep,k=3)
        
        return lowPassMap
    
    def applyHighPass(self, fwhm = 5.0):
        """                firstPower = fftTools.powerFromLiteMap(mapsAll[indicesList[i][0]],applySlepianTaper=True,\
                                              nresForSlepian = 1.0)
                lL,lU,lBin,clBinFirst, plErr_12,weights = firstPower.binInAnnuli(os.environ['LSNESRECONCSMC_DIR']+\
os.path.sep+'params/BIN_100_LOG')
        @brief Real space convolution mimicking a high pass: double laplacian and a gaussian beam at
        the high l end.
        @param FWHM FWHM of Gasussian in arcmin
        @return A highpass filtered map.
        """
        highPassMap = self.map1.takeLaplacian()
        highPassMap = highPassMap.takeLaplacian()
        #highPassMap = ltMap.convolveWithGaussian(fwhmLarge)
        #highPassMap.data[:] = ltMap.data[:] - highPassMap.data[:]
        highPassMap = highPassMap.convolveWithGaussian(fwhm)

        fwhmInRad = fwhm*numpy.pi/(180.*60.)
        fwhmLargeInRad = fwhmLarge*numpy.pi/(180.*60.)
        #self.highPassElSpaceRep = (1.0 -\
        #                           numpy.exp(-self.el**2*fwhmLargeInRad**2/(16.0*numpy.log(2.))))*\
        #                           numpy.exp(-self.el**2*fwhmInRad**2/(16.0*numpy.log(2.)))
        self.highPassElSpaceRep = (self.el**4*numpy.exp(-self.el**2*fwhmInRad**2/(16.0*numpy.log(2.))))
        self.splineHP = splrep(self.el,self.highPassElSpaceRep,k=3)
        return highPassMap
        pass
    
class  lensFiltersElSpace(lensFilters):
    """
    @brief Class derived from lensFilters for constructing lensFilters by l-space operations
    """
    def __init__(self,map1,map2,powerMap1,\
                 el,clUnlensed,noisePowermap,\
                 elObs,crossP1,crossP2,\
                 crossP3,autoPM,fwhm,\
                 cosSqCutoffParams =None):
        """
        @brief Intializes a lensFiltersElSpace instance
        @param map1 the first map (first half season)
        @param map2 the  second map (second half season)
        @param el multipole array
        @param clUnlensed unlensed theory array
        @param clNoise noise theory array
        @param elObs observed multipoles array
        @param clAutoObs1 observed auto-spectrum in map1
        @param clAutoObs2 observed auto-spectrum in map1
        @param clCrossOb observed cross-spectrum 
        @param fwhm  Beam in the experiment (arcmin, FWHM)
        @param cosSqCutoffParams tuple [lmin,lmax]
        apply a cosine-squared cutoff to the filter
        that is unity at lmin and rolls to zero at lmax
        """
        self.el = el
        self.cl = clUnlensed
        self.logSplCl = splrep(el,numpy.log(clUnlensed),k=3)
        self.noiseMap = noisePowermap
        
        self.elObs  = elObs
        #self.clAutoObs1 = clAutoObs1
        #self.logSplCl_11 = splrep(elObs,numpy.log(clAutoObs1),k=3)
        #self.SplCl_11 = splrep(elObs,clAutoObs1,k=3)
        #self.clAutoObs2 = clAutoObs2
        #self.logSplCl_22 = splrep(elObs,numpy.log(clAutoObs2),k=3)
        #self.SplCl_22 = splrep(elObs,clAutoObs2,k=3)
        #self.clCrossObs = clCrossObs
        #self.logSplCl_12 = splrep(elObs,numpy.log(clCrossObs),k=3)
        #self.SplCl_12 = splrep(elObs,clCrossObs,k=3)
        self.crossP1 = crossP1
        self.crossP2 = crossP2
        self.crossP3 = crossP3
        self.powerMap1 = powerMap1
        self.autoPM = autoPM
        #self.lowPassFilter2D = None
        #self.highPassFilter2D = None
        
        self.map1 = map1.copy()
        self.map2 = map2.copy()
        self.fwhm = fwhm
        if cosSqCutoffParams !=None:
            assert(len(cosSqCutoffParams)==2)
            
        self.cosSqCutoffParams = cosSqCutoffParams
        
    def applyLowPass(self):
        """
        @brief Applies low-pass filtering  to map2.
        """
        ft = self.map2#fftTools.fftFromLiteMap(self.map2)

        pw = prewhitener.prewhitener(1., 0.02, 0.)
        if self.cosSqCutoffParams == None:
            lowPassFilter = self.cl/(self.cl+self.nl)*0. +1.
        else:
            lmin = self.cosSqCutoffParams[0]
            lmax = self.cosSqCutoffParams[1]
        
            lowPassFilter = self.cl/(self.cl)*0.
            #lowPassFilter[250:350] = 1. - (numpy.cos((self.el[250:350]\
                #                                                      -250)*numpy.pi/\
                 #                                                    (2*(100))))**2.
            lowPassFilter[0:lmin] = self.cl[0:lmin]/(self.cl[0:lmin]+\
                                                 0.)*0. + 1.
            lowPassFilter[lmin:lmax] = \
                                     self.cl[lmin:lmax]/(self.cl[lmin:lmax]\
                                                         )\
                                                         *0. +(numpy.cos((self.el[lmin:lmax]\
                                                                      -lmin)*numpy.pi/\
                                                                     (2*(lmax-lmin))))**6.
            #deconvolving####################we are only using ls up to 20000 to prevent leakage
        ell = numpy.arange(20000)
        Filter = lowPassFilter[0:20000]#numpy.exp(ell**2.0*fwhmRad**2.0/(16.*numpy.log(2.)))
        clFilter = self.cl[0:20000]
        transferFlN = pw.correctSpectrum(self.el,(self.el*0.+1.))
        ellist = []
        filtlist = []
        #files = open('/u/sudeep/ACT/mapAnalysis/compactSources/data/B_l_AR1.dat','r')
        #for line in files:
        #    aa = line.split(' ')
        #    bb = float(aa[0])
        #    cc = 1./float(aa[1])
        #    ellist.append(bb)
        #    filtlist.append(cc)
        #print bb, cc
        #ell = ellist[0:20000]
        #Filter *= filtlist[0:20000]turned off deconv.
        print "deconvolving.."
        stripe = self.powerMap1#fftTools.powerFromLiteMap(self.map1)
        stripe.createKspaceMask(verticalStripe=[-100,100])
        block = stripe.kMask
        #block2 = stripe2.kMask

        ft2 = self.map2
        kMap = ft2.copy()
        kFilter0 = numpy.real(kMap.copy())*0.+ 1.
        kFilter0cl = numpy.real(kMap.copy())*0.+ 1.
        kFilter0nl = numpy.real(kMap.copy())*0.+ 1.
        kFilter = kMap.copy()*0.
        kclFilter = kMap.copy()*0.
        clSpline = splrep(ell,clFilter,k=3)
        #transferSpline = splrep(ell,numpy.sqrt(transferFlN[0:20000]),k=3)
        FlSpline = splrep(ell,Filter,k=3)
        ll = numpy.ravel(self.powerMap1.modLMap)
        kk = (splev(ll,FlSpline))
        kkcl = (splev(ll,clSpline))
        kFilter = numpy.reshape(kk,[self.powerMap1.Ny,self.powerMap1.Nx])
        kclFilter = numpy.reshape(kkcl,[self.powerMap1.Ny,self.powerMap1.Nx])
        self.altDims = [self.powerMap1.Ny,self.powerMap1.Nx]
        kFilter0 *= kFilter
        kFilter0cl *= kclFilter
        kFilter0nl *= self.noiseMap
        kFilterTot = kFilter*kclFilter/(kclFilter+self.noiseMap)#*block#*kFilterTr#*block2
        kMap[:,:] *= kFilter0[:,:]*kFilter0cl[:,:]/(kFilter0cl[:,:]+kFilter0nl[:,:])#*block[:,:]#*kFilterTr[:,:]#*block2[:,:]
        #ft2.kMap = kMap
        deconlensedMap2data = numpy.real(ifft2(kMap))
        ####
        #filtData = ft.mapFromFFT(kFilterFromList=[self.el,lowPassFilter])
        #self.lowPassElSpaceRep = lowPassFilter
        #self.splineLP = splrep(self.el,self.lowPassElSpaceRep,k=3)
        lowPassMap = self.map2.copy()
        #lowPassMap.data[:] = deconlensedMap2data[:]
        self.lowPassFilter2D = kFilterTot
        self.lowPassF0 = kFilterTot
        self.lowPassKMap = kMap#ft2.kMap.copy()
        plotHolder = self.powerMap1.copy()
        plotHolder.powerMap = kFilterTot.copy()
        pylab.clf()
        plotHolder.plot(zoomUptoL=3300)
        pylab.savefig('lowPassFilter.png')
        pylab.clf()
        print 'min', plotHolder.powerMap.mean()       

        del ft
        del lowPassFilter
        return lowPassMap
    
    def applyHighPass(self):

        """
        @brief  Applies high-pass filtering  to map1.
        """
        
        ft = self.map1#fftTools.fftFromLiteMap(self.map1)

        pw = prewhitener.prewhitener(1., 0.02, 0.)
        if self.cosSqCutoffParams == None:
            highPassFilter = self.cl/(self.cl+self.nl)*0. +1.
        else:
            lmin = self.cosSqCutoffParams[0]
            lmax = self.cosSqCutoffParams[1]
        
            highPassFilter = self.cl/(self.cl)*0.
            #highPassFilter[250:350] = 1. - (numpy.cos((self.el[250:350]\
             #                                                         -250)*numpy.pi/\
              #                                                       (2*(100))))**2.
            highPassFilter[0:lmin] = self.cl[0:lmin]/(self.cl[0:lmin]+\
                                                 0.)*0. + 1.#note the 200 - for low cutoff
            highPassFilter[lmin:lmax] = \
                                     self.cl[lmin:lmax]/(self.cl[lmin:lmax]\
                                                        )\
                                                         *0. +(numpy.cos((self.el[lmin:lmax]\
                                                                      -lmin)*numpy.pi/\
                                                                     (2*(lmax-lmin))))**6.
            #deconvolving####################we are only using ls up to 20000 to prevent leakage
        ell = numpy.arange(20000)
        Filter = highPassFilter[0:20000]#numpy.exp(ell**2.0*fwhmRad**2.0/(16.*numpy.log(2.)))
        clFilter = self.cl[0:20000]
        ellist = []
        filtlist = []
        #files = open('/u/sudeep/ACT/mapAnalysis/compactSources/data/B_l_AR1.dat','r')
        #for line in files:
        #    aa = line.split(' ')
        #    bb = float(aa[0])
        #    cc = 1./float(aa[1])
        #    ellist.append(bb)
        #    filtlist.append(cc)
        #transferFlN = pw.correctSpectrum(self.el,(self.el*0.+1.))
        #transferSpline = splrep(ell,(1./transferFlN[0:20000]),k=3)
        #print bb, cc
        #ell = ellist[0:20000]
        #Filter *= filtlist[0:20000] turned off deconv
        print "deconvolving.."
        stripe = self.powerMap1#fftTools.powerFromLiteMap(self.map1)
        stripe.createKspaceMask(verticalStripe=[-100,100])
        block = stripe.kMask
        #block2 = stripe2.kMask
        
        ft2 = self.map1
        kMap = ft2.copy()
        kFilter0 = numpy.real(kMap.copy())*0.+ 1.
        kFilter0cl = numpy.real(kMap.copy())*0.+ 1.
        kFilter0nl = numpy.real(kMap.copy())*0. + 1.
        kFilter = kMap.copy()*0.
        kclFilter = kMap.copy()*0.
        clSpline = splrep(ell,clFilter,k=3)
        FlSpline = splrep(ell,Filter,k=3)
        ll = numpy.ravel(self.powerMap1.modLMap)
        kk = (splev(ll,FlSpline))
        kkcl = (splev(ll,clSpline))
        #kktr = (splev(ll,transferSpline))
        #kFilterTr = numpy.reshape(kktr,[self.powerMap1.Ny,self.powerMap1.Nx])
        kFilter = numpy.reshape(kk,[self.powerMap1.Ny,self.powerMap1.Nx])
        kclFilter = numpy.reshape(kkcl,[self.powerMap1.Ny,self.powerMap1.Nx])
        kFilter0 *= kFilter
        kFilter0cl *= kclFilter
        kFilter0nl *= self.noiseMap
        kFilterTot = kFilter/(kclFilter+self.noiseMap)#*block#*block2
        kMap[:,:] *= kFilter0[:,:]/(kFilter0cl[:,:]+kFilter0nl[:,:])#*block[:,:]#*block2[:,:]

        #ft2.kMap = kMap
        deconlensedMap2data = numpy.real(ifft2(kMap))
############
        #filtData = ft.mapFromFFT(kFilterFromList=[self.el,lowPassFilter])
        #self.highPassElSpaceRep = highPassFilter
        #self.splineHP = splrep(self.el,self.highPassElSpaceRep,k=3)
        highPassMap = self.map1.copy()
        #highPassMap.data[:] = deconlensedMap2data[:]
        self.highPassFilter2D = kFilterTot
        self.highPassF0 = kFilterTot
        self.highPassKMap = kMap#ft2.kMap.copy()
        
        plotHolder2 = self.powerMap1.copy()
        plotHolder2.powerMap = kFilterTot.copy()
        pylab.clf()
        plotHolder2.plot(zoomUptoL=3300)
        pylab.savefig('highPassFilter.png')
        pylab.clf()
        print 'min', plotHolder2.powerMap.mean()
        
        del ft
        del highPassFilter
        return highPassMap

        
    
class deflectionField:
    def __init__(self):
        pass
    def generateFromKappaMap(self,kappaMap):
        maxL = numpy.floor(numpy.sqrt(numpy.pi/kappaMap.pixScaleX**2+numpy.pi/kappaMap.pixScaleY))
        el = numpy.arange(maxL)+1
        phiMap = kappaMap.filterFromList([el,2./el**2],setMeanToZero=True)
        gradm = phiMap.takeGradient()
        self.xComponent = gradm.gradX.copy()
        self.yComponent = gradm.gradY.copy()
        self.phiMap = phiMap.copy()
        del phiMap,gradm


    def plot(self,show=True):
        pass
        
    def lensMapGradApprox(self,map):
        gradMap = map.takeGradient()
        lensedMap = map.copy()
        lensedMap.data[:]  = -(gradMap.gradX.data[:]*self.xComponent.data[:] \
                              + gradMap.gradY.data[:]*self.yComponent.data[:] )
        lensedMap.data[:] += map.data[:]
        return lensedMap
        pass
    
