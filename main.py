import copy
import math
import random
from typing import *
from random import randint, random
from copy import deepcopy
from matplotlib import pyplot as plt

# dictionary of fixed and values
fixedValues: Dict[Tuple[int, int], int] = {}
# array "if greater than" with tuples of i,j where left>right
conditions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
# population size
N = 100
topPopulationPercentToCpy = 0.01
population: List[List[List[int]]] = []
matrixSize = 5
# actully mutationProbability tells us what precentage of the population should be mutate
mutationProbability = 0.3
numOfIterationsWithEqualBestFitness = 0
# we use it to normal the fitness fuction to [0,100] range
worstCaseGrade = 0.0
# keep the history of the avrage fitness in each gen
fitnessesHistory = []
# 1 for regular mode , 2 for darwin mode , and 3 for lamark mode
mode = 1


def parse(path):
    """
    should parse the file into our represntion of the game [matrixSize, fixedValues, conditions]
    :param path: path to game file
    :return: void
    """
    global matrixSize, conditions, fixedValues
    file = open(path, 'r')
    matrixSize = int(file.readline())
    numGivenDigits = int(file.readline())
    # parse given digits
    for i in range(numGivenDigits):
        digits = file.readline()
        (i, j, value) = map(lambda a: int(a), digits.split(' '))
        fixedValues[(i - 1, j - 1)] = value
    numOfConditions = int(file.readline())
    for i in range(numOfConditions):
        condition = file.readline()
        if len(condition) < 2:  # end of file with blank line
            break
        if condition[-1] == '\n':
            condition = condition[:-1]  # condition[:-1] so we dont get the \n
        (i1, j1, i2, j2) = map(lambda a: int(a) - 1, condition.split(' '))
        conditions.append(((i1, j1), (i2, j2)))


def init_population():
    """
    should create random population
    :return: void
    """
    global population
    population = []
    for k in range(N):
        currentIndividual = []
        for i in range(matrixSize):
            currentRow = []
            for j in range(matrixSize):
                if (i, j) in fixedValues.keys():
                    currentRow.append(fixedValues[(i, j)])
                else:
                    rand = randint(1, matrixSize)
                    # while rand in currentRow:
                    #     rand = randint(1, matrixSize)
                    currentRow.append(rand)
            currentIndividual.append(currentRow)
        population.append(currentIndividual)


def crossOver(first: List[List[int]], second: List[List[int]]) -> List[List[int]]:
    """
    Preform cross over.
    We choose cross point (x,y) and we take the cells of first util (x,y) cell and also we take
    the cells of second from (x,y+1) until his end, and we cross them together
    In 50% the cross over will be in random place
    :param first: the first individual
    :param second: the second individual
    :return: The cross over result
    """
    p = random()  # sometimes we want the cross to be random and some times in places where is a condition
    if p < 0.5:
        index = randint(0, len(conditions) - 1)
        (spliceRow, spliceColumn) = conditions[index][1]  # for ((3,2),(3,3)) take (3,3)
    else:
        (spliceRow, spliceColumn) = randint(0, matrixSize - 1), randint(0, matrixSize - 1)
    # preform the cross
    newIndividual = first[:spliceRow]
    newIndividual.append(first[spliceRow][:spliceColumn] + second[spliceRow][spliceColumn:])
    newIndividual += second[spliceRow + 1:]
    return newIndividual


def mutation(individual: List[List[int]]):
    """
    Every cell value can be changed by some random value, in 1/(matrixSize^2) chance
    :param individual: individual (solution to the board) to mutate
    :return: The individual after mutation
    """
    for i in range(matrixSize):
        for j in range(matrixSize):
            if random() < 1 / matrixSize ** 2:
                newNumber = randint(1, matrixSize)
                individual[i][j] = newNumber
    # another way to preform mutation
    # i = randint(0, matrixSize - 1)
    # j = randint(0, matrixSize - 1)
    # while ((i, j) not in fixedValues.keys()):
    #     i = randint(0, matrixSize - 1)
    #     j = randint(0, matrixSize - 1)
    # newNumber = randint(1, matrixSize)
    # individual[i][j] = newNumber
    return individual


def calcErrors(individual: List[List[int]]) -> float:
    """
    calculate the error grade using weights, we give a lot of weight to condition error.
    :param individual: individual/solution to the board
    :return: The error grade, the lower the better
    """
    wOfRowOrCol = 1.
    wOfCondError = 2
    w = 0.5  # give better chance to weak individuals with w < 1, give better chance to stronge individuals with w > 1.  0.5<w<0.8 seem to work
    return (numOfDuplicationInRows(individual) ** wOfRowOrCol + numOfDuplicationInColumns(individual) ** wOfRowOrCol
            + numOfUnsatefiedConditions(individual) ** wOfCondError) ** w


def calcAmountErrors(individual: List[List[int]]) -> float:
    """
    :param individual: individual
    :return: The amout of errors the board solution have
    """
    return (numOfDuplicationInRows(individual) + numOfDuplicationInColumns(individual)
            + numOfUnsatefiedConditions(individual))


def calcFitness(individual: List[List[int]]) -> float:
    """
    We use calcFitnessForOptimization to calculate the fitness.
    When we on darwin mode we want to return the fitness of the individual after optimization
    :param individual: individual/solution to the board
    :return: the individual fitness
    """
    if mode == 1 or mode == 3:
        return calcFitnessForOptimization(individual)
    if mode == 2:
        return calcFitnessForOptimization(optimization(individual))


def calcFitnessForOptimization(individual: List[List[int]]) -> float:
    """
    Using the calcErrors to compute the fitness.
    We to normalize it.
    :param individual: individual/solution to the board
    :return: the individual fitness, the higher the better (unlike calcErrors)
    """
    return 100 * (worstCaseGrade - calcErrors(individual)) / worstCaseGrade


def numOfDuplicationInRows(individual: List[List[int]]) -> int:
    """
    :param individual: individual/solution to the board
    :return: count of the number of duplication in all the rows
    for example [[3,2,2],[1,1,1],[1,2,3]] will return 1+2+0=3
    """
    numOfErrors = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            for j2 in range(j + 1, matrixSize):
                if individual[i][j] == individual[i][j2]:
                    numOfErrors += 1
    return numOfErrors


def numOfDuplicationInColumns(individual: List[List[int]]) -> int:
    """
    :param individual: individual/solution to the board
    :return: count of the number of duplication in all the columns
    for example [[3,2,2],[3,2,1],[1,2,3]] will return 1+2+0=3
    """
    numOfErrors = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            for i2 in range(i + 1, matrixSize):
                if individual[i][j] == individual[i2][j]:
                    numOfErrors += 1
    return numOfErrors


def numOfUnsatefiedConditions(individual: List[List[int]]) -> int:
    """
    :param individual: individual/solution to the board
    :return: the number of condition that were unsatefied in the current individual
    """
    numofErrors = 0
    for condition in conditions:
        pos1, pos2 = condition
        i1, j1 = pos1
        i2, j2 = pos2
        if individual[i1][j1] <= individual[i2][j2]:
            numofErrors += 1
    return numofErrors


def getTopIndividualsByPrecent() -> List[List[List[int]]]:
    """
    :return: the individuals with the best fitness
    """
    global population
    population.sort(key=calcFitness, reverse=True)
    return deepcopy(population[0:math.floor(topPopulationPercentToCpy * N)])


def createWorstIndivual():
    """
    :return: The individual with the worst possible fittness,
    It will Be the following board : [[1,1,1,1,..,1],[1,1...,1],...[1,1...,1]]
    """
    worst = []
    for i in range(N):
        curr_row = []
        for j in range(N):
            curr_row.append(1)
        worst.append(curr_row)
    return worst


def genMixRandomAndCrossedGen():
    """
    After a lot of gens we get into local max,
    So we want to take some individuals of our population, save them aside,
    create new random population and mix them together (the old and the new)
    :return: new population which built on the popultion we got so far and new random population
    """
    global population
    old = deepcopy(population)
    old.sort(key=calcFitness, reverse=True)
    init_population()
    pOfCroosed = 0.01  # the percentage we keep of out old population
    b = math.floor(N * pOfCroosed)
    population = old[:b] + population[b:]


def optimization(individual: List[List[int]]) -> List[List[int]]:
    """
    We first try to opt the condition,
    if we didnt change none, we try the opt the rows
    if we didnt change none, we try the opt the columns
    :param individual: individual/solution to the board
    :return: optimized individual
    """
    optimizedIndividual = optimizationOnConditions(individual)
    if individual != optimizedIndividual:
        return optimizedIndividual
    optimizedIndividual = optimizationOnRows(individual)
    if individual != optimizedIndividual:
        return optimizedIndividual
    optimizedIndividual = optimizationOnColumns(individual)
    if individual != optimizedIndividual:
        return optimizedIndividual
    return individual


def optimizationOnRows(individual: List[List[int]]) -> List[List[int]]:
    """
    We check for one cell in the row have value that we already have seen and we
    change her val to be val the doesnt appears int the row,
    we do that to each row.
    In the end if we get better fitness we return the opt individual,
    otherwise we stay with the same individual
    :param individual: individual/solution to the board
    :return: optimized rows individual
    """
    for i in range(matrixSize):
        remainingValues = set(range(1, matrixSize + 1))
        indexesToChange = []
        copyIndividual = copy.deepcopy(individual)
        for j in range(matrixSize):
            if (i, j) in fixedValues.keys():
                continue
            if individual[i][j] in remainingValues:
                remainingValues.remove(individual[i][j])
            else:
                indexesToChange.append(j)
        if len(indexesToChange) != 0:
            k = randint(0,len(indexesToChange))%len(indexesToChange)
            copyIndividual[i][k] = remainingValues.pop()

        if calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual):
            return copyIndividual
    return individual


def optimizationOnColumns(individual: List[List[int]]) -> List[List[int]]:
    """
    We check for one cell in the column have value that we already have seen and we
    change her val to be val the doesnt appears int the column,
    we do that to each column.
    In the end if we get better fitness we return the opt individual,
    otherwise we stay with the same individual
    :param individual: individual/solution to the board
    :return: optimized columns individual
    """
    for j in range(matrixSize):
        remainingValues = set(range(1, matrixSize + 1))
        indexesToChange = []
        copyIndividual = copy.deepcopy(individual)
        for i in range(matrixSize):
            if (i, j) in fixedValues.keys():
                continue
            if individual[i][j] in remainingValues:
                remainingValues.remove(individual[i][j])
            else:
                indexesToChange.append(j)
        if len(indexesToChange) != 0:
            k = randint(0,len(indexesToChange))%len(indexesToChange)
            copyIndividual[k][j] = remainingValues.pop()
        if calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual):
            return copyIndividual
    return individual


def optimizationOnConditions(individual: List[List[int]]) -> List[List[int]]:
    """
    We check for 3 (at most) conditions ((x,y)?(i,j)) that is unsatisfied and swipe the cells
    to ((i,j)?(x,y))
    In the end if we get better fitness we return the opt individual,
    otherwise we stay with the same individual
    :param individual: individual/solution to the board
    :return: optimized conditions individual
    """
    global conditions
    count = 3
    for (i, j), (x, y) in conditions:
        if count<=0:
            break
        copyIndividual = copy.deepcopy(individual)
        if copyIndividual[i][j] <= copyIndividual[x][y]:
            if (i, j) in fixedValues.keys():
                # can never be 1
                copyIndividual[x][y] = copyIndividual[i][j] - 1
            elif (x, i) in fixedValues.keys():
                # can never be 5
                copyIndividual[i][j] = copyIndividual[x][y] + 1
            else:
                temp = copyIndividual[i][j]
                copyIndividual[i][j] = copyIndividual[x][y]
                copyIndividual[x][y] = temp
            count=-1
        if calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual):
            return copyIndividual
    return individual


def plotGraphs():
    """
    plot the graphs in the end of the run
    :return: void
    """
    global maxFitHistory
    global averageFitHistory
    plt.plot(maxFitHistory)
    plt.title("Max fit")
    plt.ylabel("fit")
    plt.xlabel("generation")
    plt.show()
    plt.plot(averageFitHistory)
    plt.title("Average fit")
    plt.ylabel("fit")
    plt.xlabel("generation")
    plt.show()


def printGameBoard(m: List[List[int]]):
    """
    print the solution in the end of the run
    :param m:  individual/solution to the board
    :return: void
    """
    global conditions, matrixSize
    matrix = []
    try:
        for i in range(matrixSize * 2 - 1):
            row = []
            if i % 2 == 0:
                for j in range(matrixSize * 2 - 1):
                    if j % 2 == 1:
                        row.append("X")
                    if j % 2 == 0:
                        row.append(str(m[int(i / 2)][int(j / 2)]))
            else:
                for j in range(matrixSize * 2 - 1):
                    row.append("X")
            matrix.append(row)

        for (i1, j1), (i2, j2) in conditions:
            if i1 == i2:
                if j1 < j2:
                    matrix[i1 * 2][j1 * 2 + 1] = ">"
                else:
                    matrix[i1 * 2][j2 * 2 + 1] = "<"
            if j1 == j2:
                if i1 < i2:
                    matrix[i1 * 2 + 1][j1 * 2] = "v"
                else:
                    matrix[i2 * 2 + 1][j1 * 2] = "^"
        for row in matrix:
            print(row.__str__())
    except:
        print("cant print board")


if __name__ == '__main__':
    # get the file path from the user
    filePath = input('Enter your file path:')
    # get all variables from file
    parse(filePath)
    #get the mode
    mode = int(input('Enter 1 for regular mode , 2 for darwin mode , and 3 for lamark mode : '))
    # init generation
    init_population()
    worstCaseGrade = calcErrors(createWorstIndivual())
    numberOfGens = 1
    N = min(max(matrixSize ** 3, N), 500)
    numberOfGensUntilCreatingMixedGen = math.floor(N * 2)
    numberOfGensUntilStop = numberOfGensUntilCreatingMixedGen * matrixSize * matrixSize
    averageFitHistory = []
    maxFitHistory = []

    while True:
        if numberOfGens % numberOfGensUntilStop == 0: #we to many gens have pass without corrct solution, so stop
            plotGraphs()
            printGameBoard(topIndividulas[0])
            exit(0)
        elif numberOfGens % numberOfGensUntilCreatingMixedGen == 0 \
                and calcAmountErrors(population[allFitnesses.index(max(allFitnesses))]) != 0:
            #we got to local max
            genMixRandomAndCrossedGen()

        newGen = []
        allFitnesses = []
        topIndividulas = getTopIndividualsByPrecent()  # get top individuals
        for individual in population:  # calculate all fitnesses
            allFitnesses.append(calcFitness(individual))
        fitMax = max(allFitnesses)
        maxFitHistory.append(fitMax)
        numOfError = calcAmountErrors(population[allFitnesses.index(max(allFitnesses))])
        #some prints
        print("best fitness : " + str(fitMax))
        print("number of errors in best board : " + str(numOfError))
        print("num of gens : " + str(numberOfGens))

        if numOfError == 0:
            #we got the correct solution
            printGameBoard(topIndividulas[0])
            plotGraphs()
            exit(0)
        sumOfFitnesses = sum(allFitnesses)
        averageFitHistory.append(sumOfFitnesses / N)
        probabilityToBeSelected = []
        for fitness in allFitnesses:  # calculate the probability of the individual to be selected fot the cross over
            probabilityToBeSelected.append(fitness / sumOfFitnesses)

        # create the rest of the population
        #preform crossover
        for i in range(N - len(topIndividulas)):
            selectedPair = []
            for z in range(2): # choose two individuals to crossover, using their chance to be selected
                accomulatedProbability = 0
                p = random()
                for j in range(N):
                    accomulatedProbability += probabilityToBeSelected[j]
                    if accomulatedProbability >= p:
                        selectedPair.append(population[j])
                        break
            newChild = crossOver(selectedPair[0], selectedPair[1])
            newGen.append(newChild)

        # mutate over the new gen
        numToMutate = math.floor(mutationProbability * len(newGen))
        for i in range(numToMutate):
            index = randint(0, len(newGen) - 1)
            afterMutation = mutation(newGen[index])
            newGen[index] = afterMutation

        newGen += topIndividulas
        population = newGen

        # if we in lamark mode, we optimize the population
        if mode == 3:
            for i in range(N):
                population[i] = optimization(population[i])

        numberOfGens += 1
