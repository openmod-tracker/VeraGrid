# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
To run this script andes must be installed (pip install andes)
"""
import andes
import time
from andes.io.json import write
#import matplotlib
import pandas as pd
import numpy as np
#matplotlib.use('TkAgg')  # or 'QtAgg', depending on your system
from andes.utils.paths import list_cases, get_case


def main():

   start = time.time()

   ss = andes.load('Gen_Load/kundur_ieee_no_shunt.json', default_config=True)
   n_xy = len(ss.dae.xy_name)
   print(f"Andes variables = {n_xy}")
   ss.files.no_output = True

   # Run PF
   ss.PFlow.run()

   # print(f"Bus voltages = {ss.Bus.v.v}")
   # print(f"Bus angles = {ss.Bus.a.v}")

   end_pf = time.time()

   print(f"ANDES - PF time = {end_pf - start:.6f} [s]")

   # PQ constant power load
   #ss.PQ.config.p2p = 1.0
   #ss.PQ.config.p2i = 0
   #ss.PQ.config.p2z = 0
   #ss.PQ.pq2z = 0
   #ss.PQ.config.q2q = 1.0
   #ss.PQ.config.q2i = 0
   #ss.PQ.config.q2z = 0

   # Logging
   #time_history = []
   #omega_history = [[] for _ in range(len(ss.GENCLS))]
   #Ppf_history = [[] for _ in range(len(ss.PQ))]
   #tm_history = [[] for _ in range(len(ss.GENCLS))]
   #te_history = [[] for _ in range(len(ss.GENCLS))]
   #v_history = [[] for _ in range(len(ss.Bus))]
   #a_history = [[] for _ in range(len(ss.Bus))]
   #vf_history = [[] for _ in range(len(ss.GENCLS))]

   start_eig = time.time()

   eig = ss.EIG
   eig.run()

   end_eig = time.time()

   print(f"ANDES - stability - Run time: {end_eig - start_eig:.6f} [s]")
   print("State matrix A:", eig.As)
   print("State matrix A:", eig.Asc)
   print("eigenvalues:", eig.mu)
   print("Right eigenvectors:", eig.N)
   print("Left eigenvectors:", eig.W)
   print("Participation factors:", eig.pfactors)
   # print("what the hell is gyx:", eig.gyx)
   # print("Condició de A:", np.linalg.cond(eig.As))
   #print("algebraiques line:",andes.System().GENCLS.doc())


   dae = ss.dae

   return eig.As, eig.mu, eig.N, eig.W, eig.pfactors, dae.fx, dae.fy, dae.gx, dae.gy, dae.Tf, eig.gyx

if __name__ == '__main__':
    As, mu, N, W, pfactors, fx, fy, gx, gy, Tf, gyx =  main()

df_Eig = pd.DataFrame(mu)
df_Eig.to_csv("Eigenvalues_results_Andes.csv", index=False , header = False)
df_A = pd.DataFrame(As)
df_A.to_csv("A_results_Andes.csv", index=False)


# print("fx:", fx)
# print("fy:", fy)
print("gx:", gx)
# print("gy:", gy)
# print("Tf:", Tf)

sum_pf = np.array([0,0,0,0,0,0,0,0])
for col in range(pfactors.shape[0]):
    # print("col iter:",pfactors[:, col])
    sum_pf[col] = np.sum(pfactors[:, col])
# print("sum pf:", sum_pf)

#case_path = get_case('kundur/kundur_full.xlsx')
#ss = andes.run(case_path, routine='eig')
#with open('kundur_full_eig.txt', 'r') as f:
#    print(f.read())
