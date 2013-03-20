import numpy
values =  raw_input("Enter nX, nY, sizeX (deg), sizeY (deg) (separated by spaces/commas)\n")
values = values.replace(' ',',')
(nX, nY, sizeX, sizeY) = eval(values)

#Nyquist
lNyqX = 180./(1.0*sizeX)*nX
lNyqY = 180./(1.0*sizeY)*nY

lNyq = numpy.sqrt(lNyqX**2+lNyqY**2)

lfundX = 2*180./sizeX
lfundY = 2*180./sizeY

print "Maximum l in this map: %f"%lNyq

print "Fundamental frequecies in X and Y direction: %f %f"%(lfundX, lfundY)

lOne = raw_input("Enter the upper bound of the first bin (default 300) \n")
if lOne == '':
    lOne = 300.
else:
    lOne = eval(lOne)

print "The first bin has been defined as [%f,%f]"%(0.,lOne)
lUniMax = raw_input("Upto what l would you like uniform binning (default is maximum l)\n"
                    "(rest will be binned logarithmically) :\n ")
if lUniMax == '':
    lUniMax = lNyq
else:
    lUniMax = eval(lUniMax)*1.0

binW = eval(raw_input("Enter bin width for uniform binning upto %f\n"%lUniMax))*1.0

nBinsToUniMax = numpy.int(numpy.ceil((lUniMax - lOne)/binW))


binLower = lOne + 1. +numpy.arange(nBinsToUniMax)*binW
binUpper = lOne + numpy.arange(nBinsToUniMax)*binW + binW
binLower = numpy.array([0.]+ binLower.tolist())
binUpper = numpy.array([lOne] + binUpper.tolist())


if lUniMax < lNyq:
    #Logarithmic binning
    

    lUpperMax = numpy.max(binUpper)
    logWidth = numpy.log((lUpperMax+binW)/lUpperMax)
    
    nbins = numpy.ceil((numpy.log(lNyq) - numpy.log(lUpperMax))/logWidth) 
    
    xx  = numpy.arange(nbins)

    logBinLower = numpy.log(lUpperMax+1.) + xx*logWidth
    logBinUpper = numpy.log(lUpperMax) + xx*logWidth + logWidth

    lBinLower =  numpy.round(numpy.exp(logBinLower))*1.0
    lBinUpper =  numpy.round(numpy.exp(logBinUpper))*1.0

    binLower = numpy.array( binLower.tolist() + lBinLower.tolist() )
    binUpper = numpy.array( binUpper.tolist() + lBinUpper.tolist() )

binCenter = (binUpper+binLower)/2.
nBins = binLower.shape[0]
for i in xrange(nBins):
    print binLower[i] ,binUpper[i]
    

fname = raw_input("Enter output file name:")

f =open(fname, mode = "w")
f.write("%i\n"%numpy.int(nBins))
for i in xrange(nBins):
        f.write("%f %f %f\n"%(binLower[i] ,binUpper[i],binCenter[i]))
f.close()



