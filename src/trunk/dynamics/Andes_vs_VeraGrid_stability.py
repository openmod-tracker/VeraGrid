# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Eigenvalues = [2,3,4,5,6,7,8]
#df_Eig = pd.DataFrame(Eigenvalues)
#df_Eig.to_csv("Eigenvalues_results.csv", index=False)



def merge_stability_results ( csv1, csv2):
    #VeraGrid_Eig = np.loadtxt("Eigenvalues_results.csv", delimiter=",")
    #Andes_Eig = np.loadtxt("Eigenvalues_results_Andes.csv", delimiter=",")
    VeraGrid_Eig = np.genfromtxt("Eigenvalues_results.csv", delimiter=",", dtype=complex)
    Andes_Eig = np.genfromtxt("Eigenvalues_results_Andes.csv", delimiter=",", dtype=complex)

    VeraGrid_Eig_ord = VeraGrid_Eig[np.argsort(-np.abs(VeraGrid_Eig))]
    Andes_Eig_ord = Andes_Eig[np.argsort(-np.abs(Andes_Eig))]

    #print("vera eig:", VeraGrid_Eig_ord)
    #print("andes eig:", Andes_Eig_ord)
    """
    tol = 1e-10
    for e in range(len(VeraGrid_Eig_ord)):
        if abs(VeraGrid_Eig_ord[e]) < tol:
            VeraGrid_Eig_ord[e] = 1e-20
    for e in range(len(Andes_Eig_ord)):
        if abs(Andes_Eig_ord[e]) < tol:
            Andes_Eig_ord[e] = 1e-20
    

    if not len(VeraGrid_Eig_ord) == len(Andes_Eig_ord):
        VeraGrid_Eig_ord = VeraGrid_Eig_ord[:-4]
    """
    #print("vera eig ord:", VeraGrid_Eig_ord)
    #print("andes eig ord:", Andes_Eig_ord)
    print("vera eig ord abs:", np.abs(VeraGrid_Eig_ord))
    print("andes eig ord abs:", np.abs(Andes_Eig_ord))

    rel_err = np.where(np.abs(Andes_Eig_ord) != 0, np.abs(np.abs(Andes_Eig_ord) - np.abs(VeraGrid_Eig_ord)) *100/ np.abs(Andes_Eig_ord), 0)
    for i in range(len(Andes_Eig_ord)):
        if np.abs(Andes_Eig_ord[i]) <= 1e-10:
            rel_err[i] = float('inf')

    """
    rel_err = np.zeros_like(VeraGrid_Eig_ord).real.astype(float)
    print("re rel err", rel_err)
    
    """

    print("Relative error [%]:",rel_err)


    #S-domain plot
    x1 = VeraGrid_Eig_ord.real
    y1 = VeraGrid_Eig_ord.imag
    x2= Andes_Eig_ord.real
    y2 = Andes_Eig_ord.imag

    plt.scatter(x1, y1, marker='o', color='orange', label='VeraGrid')
    plt.scatter(x2, y2, marker='x', color='blue', label='Andes')
    plt.xlabel("Re [s -1]")
    plt.ylabel("Im [s -1]")
    plt.title("Stability plot")
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    plt.axhline(0, color='black', linewidth=1)  # eje horizontal (y = 0)
    plt.axvline(0, color='black', linewidth=1)
    # plt.grid(True)
    plt.legend(loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()




merge_stability_results('Eigenvalues_results.csv','Eigenvalues_results_Andes.csv')