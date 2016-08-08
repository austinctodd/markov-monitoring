#!/usr/bin/python

from math import pi, sin
from pprint import pprint
from random import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import binomial
from numpy.random import normal
from traceback import print_stack
from analytics import InfiniteUniformIntervalsSet

from  matplotlib import pyplot as plt

class FunTools(object):
    @staticmethod
    def getValsForInterval(fun, interval, binCount):
        return [fun(x) for x in np.linspace(interval[0], interval[1], binCount)]
    
    @staticmethod
    def getXsAndValsForInterval(fun, interval, binCount):
        xs = np.linspace(interval[0], interval[1], binCount)
        return xs, np.array([fun(x) for x in xs])


class FunIterEqualIntervals(object):
    def __init__(self, fun, interval, binCount):
        self.fun = fun
        self.interval = interval
        self.binCount = binCount
        self.valsIter = iter(FunTools.getValsForInterval(self.fun,
                                                         self.interval,
                                                         self.binCount))

    def __iter__(self):
        return self

    def next(self):
        return self.valsIter.next()


class FunIterBounded(object):
    def __init__(self, fun, startX, boundY, stepSize):
        self.fun = fun
        self.startX = startX
        self.boundY = boundY
        self.stepSize = stepSize
        self.currentX = None
        self.begin = True

    def __iter__(self):
        return self
    
    def next(self):
        nextX = self.startX if self.begin else self.currentX + self.stepSize
        self.begin = False
        nextY = self.fun(nextX)
        if nextY > self.boundY:
            raise StopIteration
        self.currentX = nextX
        return nextY
        

class FunIterInfinite(object):
    def __init__(self, fun, offset, stepSize):
        self.fun = fun
        self.offset = offset
        self.stepSize = stepSize
        self.nextX = self.offset
    
    def __iter__(self):
        return self
    
    def next(self):
        nextVal = self.fun(self.nextX)
        self.nextX = self.nextX + self.stepSize
    
        return nextVal

    def reset(self):
        self.nextX = self.offset


class MarkovChain(object):
    def __init__(self, offset, stepSize):
        self.offset = offset
        self.stepSize = stepSize
        self.nextVal = self.offset
    
    def __iter__(self):
        return self
    
    def next(self):
        ret = self.nextVal
        self.nextVal = self.nextVal + (random()-.5) * self.stepSize
    
        return ret


class BoundedMarkovChain(object):
    def __init__(self, offset, stepSize, lowerBound, upperBound):
        self.offset = offset
        self.stepSize = stepSize
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.begin = True
    
    def bound(self, val):
        if val <= lowerBound:
            ret = lowerBound
        elif self.offset > upperBound:
            ret = upperBound
        else:
            ret = val
        return ret

    def __iter__(self):
        return self
    
    def next(self):
        if self.begin:
            self.currentVal = self.bound(self.offset)
            self.begin = False
        else:
            self.currentVal = self.bound(
                    self.currentVal + (random()-.5) * self.stepSize)
    
        return self.currentVal


    def __init__(self, offset, stepSize, lowerBound, upperBound):
        self.offset = offset
        self.stepSize = stepSize
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.mCh = MarkovChain(offset, stepSize)
    
    def __iter__(self):
        return self
    
    def next(self):
        self.nextVal = self.mCh.next()
        if self.nextVal <= self.lowerBound:
            ret = self.lowerBound
        elif self.nextVal > self.upperBound:
            ret = self.upperBound
        else:
            ret = self.nextVal
    
        return ret

class Car(object):
    """
    sensorPdfGivenTrueVal is a function that, given a value, returns a
    sampling function.
    """
    def __init__(self, speedFun, roadIv, timeStart, timeDelta,
                 sensorPdfGivenTrueVal):
        if hasattr(speedFun, '__iter__'):
            self.speedIter = speedFun
        elif hasattr(speedFun, '__call__'):
            self.speedIter = FunIterInfinite(speedFun, timeStart, timeDelta)
        else:
            raise Exception("first arg must be either an iterator or a function")

        self.startPos = roadIv[0]
        self.endPos = roadIv[1]
        self.nextX = self.startPos
        self.timeDelta = timeDelta
        self.timeStart = timeStart
        self.nextT = timeStart
        self.sensorPdfGivenTrueVal = sensorPdfGivenTrueVal
        self.begin = True

    def __iter__(self):
        return self

    def next(self):
        if self.begin:
            self.begin = False
            return self.timeStart, self.startPos

        speed = self.speedIter.next()
        self.nextT += self.timeDelta
        self.nextX = self.nextX + speed * self.timeDelta
        if self.nextX > self.endPos:
            raise StopIteration
        return self.nextT, self.nextX

    def reset(self):
        self.begin = True
        self.speedIter.reset()
        self.nextT = self.timeStart


class OneLaneSimulation(object):
    def __init__(self, cars, pd, gpsSigma):
        self.cars = cars
        self.pd = pd
        self.simulationData = None
        self.gpsSigma = gpsSigma

    def carSensesPd(self, car):
        return [np.array([normal(x, self.gpsSigma),
                          car.sensorPdfGivenTrueVal(self.pd(x))()]) \
                for t, x in car]

    def simulate(self):
        def accumulate(a, i):
            a.extend(i)
            return a

        self.simulationData = reduce(accumulate,
                                     map(self.carSensesPd,
                                         self.cars),
                                     [])
        return self.simulationData
        
    def getSimulationData(self):
        if self.simulationData is not None:
            return self.simulationData
        self.simulate()
        return self.simulationData

    def getSimulationHist(self, iv, binCount):
        return np.histogram([p[0] for p in self.simulationData if p[1] > 0],
                            binCount,
                            iv)


class PdfGivenTrueVal(object):
    @staticmethod
    def binomial(p):
        def b(trueVal):
            if trueVal == 0:
                return lambda: binomial(1, 1-p)
            else:
                return lambda: binomial(1, p)
        return b

    @staticmethod
    def normal(sigma):
        def n(trueVal):
            return lambda: normal(trueVal, sigma)
        return n


class SecondLaneSimulation(object):
    def __init__(self, cars, pd, gpsSigma, obsCars, obsCarsSpeed):
        self.cars = cars
        self.pd = pd
        self.simulationData = None
        self.gpsSigma = gpsSigma
        self.obsCars = obsCars
        self.obsCarsSpeed = obsCarsSpeed

    def carSensesPd(self, car):
        # trueValWithObs returns the true value of the pd if there is no car
        # obstructing the view and zero otherwise. That means if the view of
        # the sensor is obstructed we still presume the same false positive
        # rate as if the view is not obstructed and there is no pd.
        def trueValWithObs(t, x):
            ###### plotting ##########
            #xs = np.linspace(car.startPos, car.endPos, car.endPos-car.startPos)
            #ys = [1 if self.obsCars.contains(xp-t*self.obsCarsSpeed) else 0\
            #        for xp in xs]
            #print "the ys plotted right now: {}".format(ys)
            #plt.plot(xs, ys, 'b-')
            #plt.axvline(x, color = 'salmon')
            #plt.show()
            ###### plotting ##########

            if self.obsCars.contains(x-t*self.obsCarsSpeed):
                return 0
            else:
                return self.pd(x)

        return [np.array([normal(x, self.gpsSigma),
                         car.sensorPdfGivenTrueVal(trueValWithObs(t, x))()]) \
                for t, x in car]

    def simulate(self):
        def accumulate(a, i):
            a.extend(i)
            return a

        self.simulationData = reduce(accumulate,
                                     map(self.carSensesPd,
                                         self.cars),
                                     [])
        return self.simulationData
        
    def getSimulationData(self):
        if self.simulationData is not None:
            return self.simulationData
        self.simulate()
        return self.simulationData


###############################################################################
def testMc():
    markov = MarkovChain(10, 1)
    mcv = []
    count = 0
    for i in markov:
        count += 1
        if count > 100:
            break
        mcv.append(i)
    
    plt.grid(True)
    plt.plot(mcv)
    plt.show()

###############################################################################

if __name__ == "__main__":
    testMc()
    #testCar()
