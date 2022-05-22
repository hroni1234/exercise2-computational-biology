import copy
import math
import sys
from typing import *
from random import randint, random
from copy import deepcopy

# locations -1
# dictionary of fixed and values
from matplotlib import pyplot as plt

fixedValues: Dict[Tuple[int, int], int] = {}
# array if greater than with tuples of i,j where left>right
conditions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
# population size
N = 100
topPopulationPercentToCpy = 0.01
population: List[List[List[int]]] = []
matrixSize = 5
mutationProbability = 0.1
numOfIterationsWithEqualBestFitness = 0
worstCaseGrade = 0.0
fitnessesHistory = []
mode=1

def parse(path):
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
    p = random()  # sometimes we want the cross to be random and some times in places where is a condition
    if p < 0.5:
        index = randint(0, len(conditions) - 1)
        (spliceRow, spliceColumn) = conditions[index][1]  # for ((3,2),(3,3)) take (3,3)
    else:
        (spliceRow, spliceColumn) = randint(0, matrixSize - 1), randint(0, matrixSize - 1)
    # (spliceRow, spliceColumn) = randint(0, matrixSize - 1), randint(0, matrixSize - 1) #totally random
    newIndividual = first[:spliceRow]
    newIndividual.append(first[spliceRow][:spliceColumn] + second[spliceRow][spliceColumn:])
    newIndividual += second[spliceRow + 1:]
    return newIndividual


def mutation(individual: List[List[int]]):
    # another way to preform mutation
    # for i in range(matrixSize):
    #     for j in range(matrixSize):
    #         if random() < 2/(N**2):
    #             newNumber = randint(1, matrixSize)
    #             individual[i][j] = newNumber
    i = randint(0, matrixSize - 1)
    j = randint(0, matrixSize - 1)
    while ((i, j) not in fixedValues.keys()):
        i = randint(0, matrixSize - 1)
        j = randint(0, matrixSize - 1)
    newNumber = randint(1, matrixSize)
    individual[i][j] = newNumber
    return individual


def calcErrors(individual: List[List[int]]) -> float:
    wOfRowOrCol = 1.
    wOfCondError = 1.25
    w = 0.75  # give better chance to weak individuals with w < 1, give better chance to stronge individuals with w > 1.  0.5<w<0.8 seem to work
    return (numOfDuplicationInRows(individual) ** wOfRowOrCol + numOfDuplicationInColumns(individual) ** wOfRowOrCol
            + numOfUnsatefiedConditions(individual) ** wOfCondError) ** w


def calcAmountErrors(individual: List[List[int]]) -> float:
    return (numOfDuplicationInRows(individual) + numOfDuplicationInColumns(individual)
            + numOfUnsatefiedConditions(individual))

def calcFitness(individual: List[List[int]]) -> float:
    if(mode==1 or mode==3):
        return calcFitnessForOptimization(individual)
    if(mode==2):
        return calcFitnessForOptimization(optimization(individual));

def calcFitnessForOptimization(individual: List[List[int]]) -> float:
    return 100 * (worstCaseGrade - calcErrors(individual)) / worstCaseGrade

def numOfDuplicationInRows(individual: List[List[int]]) -> int:
    numOfErrors = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            for j2 in range(j + 1, matrixSize):
                if individual[i][j] == individual[i][j2]:
                    numOfErrors += 1
    return numOfErrors


def numOfDuplicationInColumns(individual: List[List[int]]) -> int:
    numOfErrors = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            for i2 in range(i + 1, matrixSize):
                if individual[i][j] == individual[i2][j]:
                    numOfErrors += 1
    return numOfErrors


def numOfUnsatefiedConditions(individual: List[List[int]]) -> int:
    numofErrors = 0
    for condition in conditions:
        pos1, pos2 = condition
        i1, j1 = pos1
        i2, j2 = pos2
        if individual[i1][j1] < individual[i2][j2]:
            numofErrors += 1
    return numofErrors


def getTopIndividualsByPrecent() -> List[List[List[int]]]:
    global population
    population.sort(key=calcFitness, reverse=True)
    return deepcopy(population[0:math.floor(topPopulationPercentToCpy * N)])


def createWorstIndivual():
    worst = []
    for i in range(N):
        curr_row = []
        for j in range(N):
            curr_row.append(1)
        worst.append(curr_row)
    return worst


def genMixRandomAndCrossedGen():
    global population
    old = deepcopy(population)
    old.sort(key=calcFitness, reverse=True)
    init_population()
    pOfCroosed = 0.05
    b = math.floor(N * pOfCroosed)
    population = old[:b] + population[b:]

def optimization(individual: List[List[int]])->List[List[int]]:
    optimizedIndividual = optimizationOnConditions(individual);
    if (individual != optimizedIndividual):
        return optimizedIndividual
    optimizedIndividual = optimizationOnRows(individual);
    if(individual!= optimizedIndividual):
        return optimizedIndividual
    optimizedIndividual = optimizationOnColumns(individual);
    if (individual != optimizedIndividual):
        return optimizedIndividual
    return individual

def optimizationOnRows(individual: List[List[int]])->List[List[int]]:
    for i in range(matrixSize):
        remainingValues = set(range(1,matrixSize+1));
        indexesToChange =[];
        copyIndividual = copy.deepcopy(individual);
        for j in range(matrixSize):
            if ((i,j) in fixedValues.keys()):
                continue;
            if(individual[i][j] in remainingValues):
                remainingValues.remove(individual[i][j])
            else:
                indexesToChange.append(j);
        for k in indexesToChange:
            copyIndividual[i][k]=remainingValues.pop();
        if(calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual)):
            return copyIndividual;
    return individual;


def optimizationOnColumns(individual: List[List[int]])->List[List[int]]:
    for j in range(matrixSize):
        remainingValues = set(range(1,matrixSize+1));
        indexesToChange = [];
        copyIndividual = copy.deepcopy(individual);
        for i in range(matrixSize):
            if ((i,j) in fixedValues.keys()):
                continue;
            if (individual[i][j] in remainingValues):
                remainingValues.remove(individual[i][j])
            else:
                indexesToChange.append(j);
        for k in indexesToChange:
            copyIndividual[k][j] = remainingValues.pop();
        if (calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual)):
            return copyIndividual;
    return individual;

def optimizationOnConditions(individual: List[List[int]])->List[List[int]]:
    for (i,j), (x,y) in conditions:
        copyIndividual = copy.deepcopy(individual);
        if(copyIndividual[i][j]<= copyIndividual[x][y]):
            if ((i,j) in fixedValues.keys()):
                #can never be 1
                copyIndividual[x][y]=copyIndividual[i][j]-1;
            elif ((x,i) in fixedValues.keys()):
                #can never be 5
                copyIndividual[i][j]=copyIndividual[x][y]+1;
            else :
                temp= copyIndividual[i][j]
                copyIndividual[i][j]=copyIndividual[x][y]
                copyIndividual[x][y]=temp
        if (calcFitnessForOptimization(individual) < calcFitnessForOptimization(copyIndividual)):
            return copyIndividual;
    return individual;

def plotGraphs():
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

def printGraph(m:List[List[int]]):
    global conditions, matrixSize
    matrix = []
    for i in range(matrixSize*2):
        row = []
        if i%2==0:
            for j in range(matrixSize*2):
                if j%2 == 1:
                    row.append("X")
                if j%2 == 0:
                    row.append(str(m[int(i/2)][int(j/2)]))
        else:
            for j in range(matrixSize * 2):
                row.append("X")
        matrix.append(row)

    for (i1,j1), (i2,j2) in conditions:
        if i1 == i2:
            matrix[i1*2][j2*2-1] = ">"
        if j1 == j2:
            matrix[i2*2-1][j1*2] ="v"
    for row in matrix:
        print(row)

if __name__ == '__main__':
    #get the file path from the user
    filePath = input('Enter your file path:')
    # get all variables from file
    parse(filePath)
    mode = int(input('Enter 1 for regular mode , 2 for darwin mode , and 3 for lamark mode : '))
    # init generation
    init_population()
    worstCaseGrade = calcErrors(createWorstIndivual())
    numberOfGens = 1
    N = min(max(matrixSize ** 3, N), 500)
    numberOfGensUntilCreatingMixedGen = math.floor(N * 2)
    numberOfGensUntilStop = numberOfGensUntilCreatingMixedGen*matrixSize
    averageFitHistory = []
    maxFitHistory = []

    while (True):
        if numberOfGens % numberOfGensUntilStop == 0:
            plotGraphs()
            printGraph(topIndividulas[0])
            exit(0)
        elif numberOfGens % numberOfGensUntilCreatingMixedGen == 0 \
                and calcAmountErrors(population[allFitnesses.index(max(allFitnesses))]) != 0:
            genMixRandomAndCrossedGen()
        newGen = []
        allFitnesses = []
        topIndividulas = getTopIndividualsByPrecent()  # get top individuals
        for individual in population:  # calculate all fitnesses
            allFitnesses.append(calcFitness(individual))
        fitMax = max(allFitnesses)
        maxFitHistory.append(fitMax)
        print(fitMax)
        if fitMax == 100 and calcAmountErrors(topIndividulas[0]) == 0:
            print(topIndividulas[0])
            printGraph(topIndividulas[0])
            plotGraphs()
            exit(0)
        # print(population[allFitnesses.index(max(allFitnesses))])
        print(calcAmountErrors(population[allFitnesses.index(max(allFitnesses))]))
        print("num of gens : " + str(numberOfGens))
        sumOfFitnesses = sum(allFitnesses)
        averageFitHistory.append(sumOfFitnesses/N)

        probabilityToBeSelected = []
        for fitness in allFitnesses:  # calculate the probability of the fitness
            probabilityToBeSelected.append(fitness / sumOfFitnesses)

        # create the rest of the population
        for i in range(N - len(topIndividulas)):
            selectedPair = []
            for z in range(2):
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
        numberOfGens += 1
        printGraph(topIndividulas[0])
        if(mode==3):
            for i in range(N):
                population[i]=optimization(population[i])