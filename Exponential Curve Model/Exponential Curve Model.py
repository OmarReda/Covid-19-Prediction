
""" VERY IMPORTMANT """
""" 
Before run 


-choose which data to experiment on by uncommenting either

File_Path = dir_path + "/Cairo.csv" #line 48
or 
File_Path = dir_path + "/NewYork.csv" #line 49

to use the data 


-choose either daily or total case by uncomemnting either

a, b, c = obj.ComputeCoeff(temperature, humidity, cases) #line 164
or
a, b, c = obj.ComputeCoeff(Avg_Temp,Growth_Rate,total_cases) #line 165


-also uncomment either 

plt.plot(days, cases, "blue") #line 63
or
plt.plot(days, total_cases, "blue") #line 64

to plot daily cases or total cases ( original data)
"""

import numpy as np
from sympy import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, diff
from numpy.linalg import inv
import os


class ModelOne(object):
    def ReadData(self):
        # Insert File path here
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Chosse Wether you want to display cairo or Newyork results ,uncomment only  1 of them

        #File_Path = dir_path + "/Cairo.csv" #Uncomment to use cairo data
        #File_Path = dir_path + "/NewYork.csv" #Uncomment to use Newyork data


        df = pd.read_csv(File_Path)
        days, temperature, humidity, cases,Avg_Temp,Growth_Rate,total_cases = (
            df["NumberOfDays"],
            df["Temp"].values,
            df["Humidity"].values,
            df["Daily_cases"].values,
            df["Avg_Temp"].values,
            df["Growth_Rate"].values,
            df["Total"].values
        )
        plt.figure("Original Data")
        #plt.plot(days, cases, "blue") #uncomment to show data based on daily case
        #plt.plot(days, total_cases, "blue") #uncomment to show data based on total cases 
        plt.ylabel("Cases")
        plt.xlabel("NumberOfDays")
        plt.show()
        return days, temperature, humidity, cases,Avg_Temp,Growth_Rate,total_cases

    def ComputeCoeff(self, x1, x2, y):

        a, b, c = symbols("a b c", real=True)

        F = [0] * 3  # Function Matrix (number of functions)

        length = len(F)
        jacabMatrix = [[0 for x in range(length)] for y in range(length)]

        # Defining Functions

        mainFunc = (y[0] - a * (b ** x1[0]) * (c ** x2[0])) ** 2

        for i in range(1, 60):
            mainFunc = mainFunc + (y[i] - a * (b ** x1[i]) * (c ** x2[i])) ** 2

        F[0] = diff(mainFunc, a)
        # Diff by b
        F[1] = diff(mainFunc, b)
        # Diff by c
        F[2] = diff(mainFunc, c)

        for i in range(0, len(F)):
            # Diff by a
            jacabMatrix[i][0] = diff(F[i], a)
            # Diff by b
            jacabMatrix[i][1] = diff(F[i], b)
            # Diff by c
            jacabMatrix[i][2] = diff(F[i], c)

        # newton
        iterations = 0
        jacabMatrixVal = [[0 for x in range(length)] for y in range(length)]
        FVal = [0] * length

        root = [1.5, 1.5, 1.5]  # Intial matrix
        err = 100

        while err > (10 ** (-2)):

            print("*******************************************")
            print("Iteration", iterations)
            print("*******************************************")

            # get the old root
            old_root = np.array(root).astype(
                np.float64
            )  # you may need to convert root matrix to list

            # Compute Hessian Matrix
            for i in range(0, length):
                for j in range(0, length):
                    jacabMatrixVal[i][j] = jacabMatrix[i][j].subs(
                        [(a, root[0]), (b, root[1]), (c, root[2])]
                    )

            # Compute Function Matrix
            for i in range(0, length):
                FVal[i] = F[i].subs([(a, root[0]), (b, root[1]), (c, root[2])])

            # Convert lists to matrices
            jacabMatrixVal = np.asarray(jacabMatrixVal)
            FVal = np.asarray(FVal)

            # compute Hessian Inverse
            jacabMatrixInv = inv(np.matrix(jacabMatrixVal, dtype="float"))

            # Compute new roots
            root = root - jacabMatrixInv.dot(FVal)

            # break at 100 iteration

            root_norm = np.array(root).astype(np.float64)

            # compute Error
            err = abs(np.linalg.norm(root_norm - old_root) / np.linalg.norm(old_root))
            root = root.tolist()[0]
            print("Result root : ", root)
            print("err : ", err)
            iterations = iterations + 1
            # Break at 10 iterations
            if iterations == 10:
                break

        return root[0], root[1], root[2]


def main():
    x1, x2 = symbols("x1 x2", real=True)

    obj = ModelOne()
    # Read dataset
    days, temperature, humidity, cases,Avg_Temp,Growth_Rate,total_cases = obj.ReadData()
    # Compute Coeff.s
    #a, b, c = obj.ComputeCoeff(temperature, humidity, cases) #Uncomment To test Daily cases
    #a, b, c = obj.ComputeCoeff(Avg_Temp,Growth_Rate,total_cases) #Uncomment To test total cases
    print("Roots A= ", a, " B=", b, " C= ", c)

    mainFunc = a * (b ** x1) * (c ** x2)

    Predicted_Cases = []
    for i in range(0, len(temperature)):
        predicition = int(mainFunc.subs([(x1, temperature[i]), (x2, humidity[i])]))
        Predicted_Cases.append(predicition)

    plt.figure("Predicted Data")
    plt.plot(days, Predicted_Cases, "blue")
    plt.ylabel("Cases")
    plt.xlabel("NumberOfDays")
    plt.show()

    pass


if __name__ == "__main__":
    main()
