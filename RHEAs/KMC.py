'''
    KMC计算H在高熵合金中扩散系数
'''

import numpy as np
import random
import math
import matplotlib.pyplot as plt


def neighborlist(system):
    # Read in atomic positions of the metal matrix (perfect crystal)
    fname = 'POSCAR'
    fhand = open(fname)
    header = []
    atoms = []
    for n, line in enumerate(fhand):
        if n <= 7:
            header.append(line)
        if n > 7:
            line = line.rstrip().split()
            if len(line) == 3:
                atom = [float(line[0]), float(line[1]), float(line[2])]
                atoms.append(atom)
    lattice_const = float(header[1].rstrip().split()[0])
    xhi = float(header[2].rstrip().split()[0])
    yhi = float(header[3].rstrip().split()[1])
    zhi = float(header[4].rstrip().split()[2])
    for atom in atoms:
        if atom[0] < 0.3:
            atoms.append([atom[0] + 1, atom[1], atom[2]])
        if atom[1] < 0.3:
            atoms.append([atom[0], atom[1] + 1, atom[2]])
        if atom[2] < 0.3:
            atoms.append([atom[0], atom[1], atom[2] + 1])

    for atom in atoms:
        if atom[0] > 0.7:
            atoms.append([atom[0] - 1, atom[1], atom[2]])
        if atom[1] > 0.7:
            atoms.append([atom[0], atom[1] - 1, atom[2]])
        if atom[2] > 0.7:
            atoms.append([atom[0], atom[1], atom[2] - 1])
    atoms = np.array(atoms)
    for atom in atoms:
        atom[0] = round(atom[0] * xhi * lattice_const, 3)
        atom[1] = round(atom[1] * yhi * lattice_const, 3)
        atom[2] = round(atom[2] * zhi * lattice_const, 3)
    atoms = np.unique(atoms, axis=0)

    fname = 'H_positions.dat'  # POSCAR file, Direct coordinate, lattice constant on the second row
    fhand = open(fname)
    H_atoms = []
    for n, line in enumerate(fhand):
        line = line.rstrip().split()
        if len(line) == 3:
            atom = [float(line[0]), float(line[1]), float(line[2]), n]
            H_atoms.append(atom)

    for atom in H_atoms:
        if atom[0] < 0.3:
            H_atoms.append([atom[0] + 1, atom[1], atom[2], atom[3]])
        if atom[1] < 0.3:
            H_atoms.append([atom[0], atom[1] + 1, atom[2], atom[3]])
        if atom[2] < 0.3:
            H_atoms.append([atom[0], atom[1], atom[2] + 1, atom[3]])

    for atom in H_atoms:
        if atom[0] > 0.7:
            H_atoms.append([atom[0] - 1, atom[1], atom[2], atom[3]])
        if atom[1] > 0.7:
            H_atoms.append([atom[0], atom[1] - 1, atom[2], atom[3]])
        if atom[2] > 0.7:
            H_atoms.append([atom[0], atom[1], atom[2] - 1, atom[3]])
    H_atoms = np.array(H_atoms)
    for atom in H_atoms:
        atom[0] = round(atom[0] * xhi * lattice_const, 3)
        atom[1] = round(atom[1] * yhi * lattice_const, 3)
        atom[2] = round(atom[2] * zhi * lattice_const, 3)
    H_atoms = np.unique(H_atoms, axis=0)

    # classify the H positions into saddle points, TIs and OIs based on their numbers of neighboring metal atoms
    saddles = []
    for H_atom in H_atoms:
        neighbors = []
        for atom in atoms:
            if cal_distance(H_atom, atom) < 2:
                neighbors.append(atom)
        if len(neighbors) == 3:  # if the H atom has 3 nearest neighbors, then it should be at a saddle point
            saddles.append(H_atom)

    TIs = []
    for H_atom in H_atoms:
        neighbors = []
        for atom in atoms:
            if cal_distance(H_atom, atom) < 2:
                neighbors.append(atom)
        if len(neighbors) == 4:  # if the H atom has 4 nearest neighbors, then it should be at a TI
            TIs.append(H_atom)

    OIs = []
    for H_atom in H_atoms:
        neighbors = []
        for atom in atoms:
            if cal_distance(H_atom, atom) < 2:
                neighbors.append(atom)
        if len(neighbors) == 6:  # if the H atom has 6 nearest neighbors, then it should be at a OI
            OIs.append(H_atom)
    H_num = 0
    H_atom_inbox = []
    for H_atom in H_atoms:
        if In_Boundary(H_atom, xhi * lattice_const, yhi * lattice_const,
                       zhi * lattice_const):  # exclude atoms outside the simulation box
            H_num = H_num + 1
            H_atom_inbox.append(H_atom)

    # build the neighbor list based on the connectivity of the TIs, OIs and saddle points
    OI_TI_saddle = []
    for OI in OIs:
        if In_Boundary(OI, xhi * lattice_const, yhi * lattice_const, zhi * lattice_const):
            neighbor_OI = []
            neighbor_saddle = []
            TI_saddle = []
            for TI in TIs:
                if cal_distance(OI, TI) < 1.8:
                    neighbor_OI.append(TI)
            for TI in neighbor_OI:
                nearest_saddle = []
                for saddle in saddles:
                    if cal_distance(saddle, TI) < 0.8:
                        nearest_saddle.append(saddle)
                for saddle in nearest_saddle:
                    if cal_distance(OI, saddle) < cal_distance(OI, TI):
                        neighbor_saddle.append(saddle)
                        TI_saddle.append([OI, TI, saddle])
            OI_TI_saddle.append(TI_saddle)

    TI_OI_saddle = []
    for TI in TIs:
        if In_Boundary(TI, xhi * lattice_const, yhi * lattice_const, zhi * lattice_const):
            neighbor_TI = []
            neighbor_saddle = []
            OI_saddle = []
            for saddle in saddles:
                if cal_distance(saddle, TI) < 0.8:
                    neighbor_saddle.append(saddle)
            for OI in OIs:
                if cal_distance(OI, TI) < 1.8:
                    neighbor_TI.append(OI)
            for OI in neighbor_TI:
                for saddle in neighbor_saddle:
                    if cal_distance(OI, saddle) < cal_distance(OI, TI):
                        OI_saddle.append([TI, OI, saddle])
            TI_OI_saddle.append(OI_saddle)

    neighborlist = [OI_TI_saddle, TI_OI_saddle]


    return (neighborlist)


def kMC(neighborlist, sol_energy):  # input the neighbor list and the H solution energies
    OI_TI_saddle = neighborlist[0]
    TI_OI_saddle = neighborlist[1]
    # kMC steps start
    kBT = 0.025852
    t = []
    distance = []
    OI_id = random.randint(0, len(OI_TI_saddle) - 1)
    positions = []
    for step in range(100000):
        rates = []
        ##### OI jump to TI##############################################
        for TI_saddle in OI_TI_saddle[OI_id]:
            OI = TI_saddle[0]
            TI = TI_saddle[1]
            saddle = TI_saddle[2]
            barrier = sol_energy[int(saddle[-1])] - sol_energy[int(OI[-1])]
            rate = 10 ** 13 * math.exp(-barrier / kBT)
            rates.append(rate)
        sum_rates = sum(rates)
        x = [0]
        accum_rate = 0
        for rate in rates:
            accum_rate = accum_rate + rate
            x.append(accum_rate / sum_rates)

        gamma = random.random()
        for i in range(0, len(x)):
            if gamma > x[i] and gamma < x[i + 1]:
                next_TI_index = i
        next_TI = OI_TI_saddle[OI_id][next_TI_index][1]
        positions.append(next_TI)
        rho = random.random()
        t.append(-math.log(rho) / sum_rates)
        distance.append(cal_distance(next_TI, OI))

        ##### TI jump to OI##############################################
        for i, OI_saddle in enumerate(TI_OI_saddle):
            TI = OI_saddle[0][0]
            if TI[-1] == next_TI[-1]:
                next_TI_ID = i
        rates = []
        for OI_saddle in TI_OI_saddle[next_TI_ID]:
            TI = OI_saddle[0]
            OI = OI_saddle[1]
            saddle = OI_saddle[2]
            barrier = sol_energy[int(saddle[-1])] - sol_energy[int(TI[-1])]
            rate = 10 ** 13 * math.exp(-barrier / kBT)
            rates.append(rate)
        sum_rates = sum(rates)
        x = [0]
        accum_rate = 0
        for rate in rates:
            accum_rate = accum_rate + rate
            x.append(accum_rate / sum_rates)

        gamma = random.random()
        for i in range(0, len(x)):
            if gamma > x[i] and gamma < x[i + 1]:
                next_OI_index = i
        next_OI = TI_OI_saddle[next_TI_ID][next_OI_index][1]
        positions.append(next_OI)
        rho = random.random()
        t.append(-math.log(rho) / sum_rates)
        distance.append(cal_distance(next_OI, TI_OI_saddle[next_TI_ID][0][0]))

        for i, TI_saddle in enumerate(OI_TI_saddle):
            OI = TI_saddle[0][0]
            if OI[-1] == next_OI[-1]:
                next_OI_ID = i
        OI_id = next_OI_ID
    positions = np.array(positions)

    D = 0
    t_add = 0
    add_D = []
    add_t = []
    for step in range(len(t)):
        delta_D = distance[step] ** 2
        D = D + delta_D
        t_add = t_add + t[step]
        add_D.append(D)
        add_t.append(t_add)
    d_D = add_D[-1] - add_D[0]
    d_t = add_t[-1] - add_t[0]
    D_coeff = d_D / (6 * d_t) * 10e-20  # convert unit to m2/s
    plt.plot(add_t, add_D)
    return D_coeff

