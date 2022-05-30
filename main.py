import numpy as np
import numpy.random
import matplotlib.pyplot as plt

MU = 1  # Since mu
SECOND_EPSILON = 0.0001
NUM_OF_CONSTS = 500
LOW = 0.01
HIGH = 20
NUM_OF_MATRICES = 50000
MAX_MATRIX_VAL = 100
MIN_DET = 0.25
MAX_DET = 50
EPSILON = 0.001
REG_DIV_CONST = 1 / 77
REG_DIV_INT = 75


def genCiarletFunction(gammaPrime, gammaPrimeTwice):
    a = MU + (0.5 * gammaPrime)
    b = -0.5 * (MU + gammaPrime)
    c = 0.25 * (gammaPrime + gammaPrimeTwice)
    d = 0.5 * (gammaPrimeTwice - gammaPrime)

    def func(F):
        xSquared = np.square(F[0])
        ySquared = np.square(F[1])
        zSquared = np.square(F[2])
        firstTerm = a * (xSquared + ySquared + zSquared - 3)
        secondTerm = b * ((xSquared * ySquared) + (ySquared * zSquared) + (xSquared * zSquared) - 3)
        thirdTerm = c * ((xSquared * ySquared * zSquared) - 1)
        fourthTerm = d * (np.log(F[0]) + np.log(F[1]) + np.log(F[2]))
        return firstTerm + secondTerm + thirdTerm - fourthTerm

    return func


def getRandomConstants():
    randLambda = np.zeros(NUM_OF_CONSTS)
    gamma = np.zeros((NUM_OF_CONSTS, 2))
    # creating 2 arrays to hold gamma' and gamma''
    gammaDPrimeArr = np.zeros((NUM_OF_CONSTS, REG_DIV_INT))
    gammaPrimeArr = np.zeros((NUM_OF_CONSTS, REG_DIV_INT))
    # randomize lambda between 2mu and 0
    for i in range(NUM_OF_CONSTS):  # TODO: lambda to lambda + mu is the other one, then no need for gammaDPrime > gammaPrim since it always happens
        randLambda[i] = np.random.uniform(LOW, 2)
        gamma[i, 1] = numpy.random.uniform(0.5 * randLambda[i] + MU + SECOND_EPSILON, randLambda[i] + 1)
        gammaDPrimeArr[i] = np.arange(REG_DIV_CONST, MU - REG_DIV_CONST, REG_DIV_CONST, dtype=float)
        gammaDPrimeArr[i] = gammaDPrimeArr[i] * (randLambda[i] - (0.5 * randLambda[i] + SECOND_EPSILON))
        gammaDPrimeArr[i] = gammaDPrimeArr[i] + 0.5 * randLambda[i] + MU + SECOND_EPSILON
        gammaPrimeArr[i] = gammaDPrimeArr[i] - randLambda[i] - 2
    return randLambda, gammaPrimeArr, gammaDPrimeArr


def genRandomMatrices():
    randMatrices = numpy.random.rand(NUM_OF_MATRICES, 3) * MAX_MATRIX_VAL
    for ind in range(NUM_OF_MATRICES):
        curDet = randMatrices[ind][0] * randMatrices[ind][1] * randMatrices[ind][2]
        while curDet == 0:
            randMatrices[ind] = numpy.random.rand(3) * MAX_MATRIX_VAL
            curDet = randMatrices[ind][0] * randMatrices[ind][1] * randMatrices[ind][2]
        if curDet < MIN_DET or curDet > MAX_DET:
            scalar = (1 / curDet) * np.random.uniform(MIN_DET, MAX_DET)
            randMatrices[ind] *= numpy.cbrt(scalar)
    return randMatrices


def main():
    lambdaArr, gammaPrime, gammaDPrime = getRandomConstants()
    functionArr = []
    for i in range(NUM_OF_CONSTS):
        functionArr.append([])
        for j in range(REG_DIV_INT):
            functionArr[i].append(genCiarletFunction(gammaPrime[i][j], gammaDPrime[i][j]))
    matrices = genRandomMatrices()
    posX = np.zeros(shape=(NUM_OF_CONSTS, REG_DIV_INT))
    posY = np.zeros(shape=(NUM_OF_CONSTS, REG_DIV_INT))
    notPosX = np.zeros(shape=(NUM_OF_CONSTS, REG_DIV_INT))
    notPosY = np.zeros(shape=(NUM_OF_CONSTS, REG_DIV_INT))
    for muLambda in range(NUM_OF_CONSTS):
        print(muLambda)
        for gamma in range(REG_DIV_INT):
            isPos = True
            for matrixInd in range(NUM_OF_MATRICES):
                val = functionArr[muLambda][gamma](matrices[matrixInd])
                if val <= EPSILON:
                    isPos = False
                    notPosX[muLambda, gamma] = lambdaArr[muLambda]
                    notPosY[muLambda, gamma] = gammaDPrime[muLambda][gamma]
                    break
            if isPos:
                posX[muLambda, gamma] = lambdaArr[muLambda]
                posY[muLambda, gamma] = gammaDPrime[muLambda][gamma]

    plt.scatter(posX.flatten(), posY.flatten(), label="positive")
    plt.scatter(notPosX.flatten(), notPosY.flatten(), label="not positive")

    plt.legend()
    plt.show()
    plt.savefig("functions_graph")
    return


if __name__ == '__main__':
    main()
