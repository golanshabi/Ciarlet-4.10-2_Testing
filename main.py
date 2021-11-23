import numpy as np
import numpy.linalg as lg
import numpy.random

NUM_OF_CONSTS = 10
LOW = 0.01
HIGH = 100
NUM_OF_MATRICES = 10000
MAX_MATRIX_VAL = 100
MIN_DET = 0.25
MAX_DET = 50
EPSILON = 0.1


def genCiarletFunction(mu, gammaPrime, gammaPrimeTwice):
    a = mu + 0.5 * gammaPrime
    b = -0.5 * (mu + gammaPrime)
    c = 0.25 * (gammaPrime + gammaPrimeTwice)
    d = 0.5 * (gammaPrimeTwice - gammaPrime)
    e = - (3 * a + 3 * b + c)

    def func(F, detF):
        norm = np.square(lg.norm(F))
        return a * norm + b * (
                    0.5 * norm - 0.5 * np.square(lg.norm(lg.matrix_power(F * F.T, 2)))) + c * np.square(detF) -\
        d * np.log(detF) + e

    return func


def getRandomConstants():
    randLame = np.random.uniform(LOW, HIGH, (NUM_OF_CONSTS, 2))
    gamma = np.array(NUM_OF_CONSTS, 2)
    for i in range(NUM_OF_CONSTS):
        gamma[i][0] = numpy.random.uniform(-(randLame[i][1] + randLame[i][0]), -randLame[i][0])
        gamma[i][1] = numpy.random.uniform(randLame[i][1] + randLame[i][0], randLame[i][1] + randLame[i][0])
    return randLame, gamma


def genRandomMatrices():
    randMatrices = (numpy.random.rand(NUM_OF_MATRICES, 3, 3) - 0.5) * MAX_MATRIX_VAL
    dets = np.array(NUM_OF_MATRICES)
    for ind in range(NUM_OF_MATRICES):
        curDet = lg.det(randMatrices[ind])
        while curDet == 0:
            randMatrices[ind] = (numpy.random.rand(3, 3) - 0.5) * MAX_MATRIX_VAL
            curDet = lg.det(randMatrices[ind])
        if curDet < MIN_DET or curDet > MAX_DET:
            scalar = (1 / curDet) * np.random.uniform(MIN_DET, MAX_DET)
            randMatrices[ind][0] *= scalar
        dets[ind] = curDet
    return randMatrices, dets


def main():
    lame, gamma = getRandomConstants()
    functionArr = []
    for i in range(NUM_OF_CONSTS):
        functionArr.append(genCiarletFunction(lame[i][1], gamma[i][0], gamma[i][1]))
    matrices, determinants = genRandomMatrices()
    for matrixInd in range(NUM_OF_MATRICES):
        for ciarletInd in range(NUM_OF_CONSTS):
            val = functionArr[ciarletInd](matrices[matrixInd], determinants[matrixInd])
            if val < EPSILON:
                print("-------------------------------------------------------")
                print("matrix is: ")
                print(matrices[matrixInd])
                print("val is: ")
                print(val)
                print("mu is: ")
                print(lame[ciarletInd][1])
                print("lambda is: ")
                print(lame[ciarletInd][0])
                print("gammaPrime is: ")
                print(gamma[ciarletInd][0])
                print("gammaPrimeTwice is: ")
                print(gamma[ciarletInd][1])
    return


if __name__ == '__main__':
    main()
