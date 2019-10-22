# ModeMap.py


JobDirPrefix = "ModeMap-Band01"

RunVASP = r"mpirun -ppn 36 -n 36 vasp_std"

PhononPOSCAR = r"POSCAR.Opt"
PhononOUTCAR = r"OUTCAR.Phonon"

BandIndex = -1

QMin = -1.50
QMax =  1.50
QInc =  0.10


import math
import glob
import os
import re
import shutil
import sys

import numpy as np


def _CheckComplete(file_path = r"OUTCAR"):
    """
    Inspects an OUTCAR file and (crudely!) determines whether a VASP job has finished cleanly.

    Params:
       file_path -- path to an OUTCAR file (default: OUTCAR)

    Returns:
       Number of ionic steps N if the calculation completed, or -1 otherwise.
    """

    # If file_path does not exist, assume VASP failed to start.

    if not os.path.isfile(file_path):
        return -1

    num_steps = 0

    if os.path.isfile(file_path):
        with open(file_path, 'r') as input_reader:
            for line in input_reader:
               if "LOOP+" in line:
                   # Identify ionic steps by the "LOOP+" string.

                   num_steps += 1

               elif "Voluntary context switches" in line:
                   # Unless a user does something strange this string _should_ only appear in the last line of the OUTCAR.

                   return num_steps

    return -1

def _ReadE0(file_path = r"OSZICAR"):
    """
    Read an OSZICAR file and return the total energy E0 from the last electronic SCF step.

    Params:
      file_path -- path to an OSZICAR file (default: OSZICAR)

    Returns:
      Total energy E0 on success, or None on failure.
    """

    e0_regex = re.compile(r"E0=\s*(?P<e0>[+-]?\d*\.\d+E[+-]?\d+)")

    e0 = None

    with open(file_path, 'r') as input_reader:
        for line in input_reader:
            match = e0_regex.search(line)

            if match:
                e0 = float(
                    match.group('e0')
                    )

    return e0

def _ReadStructure(file_path = r"POSCAR"):
    """
    Read a POSCAR file and return the lattice vectors, atom types and positions.

    Params:
        file_path -- path to a POSCAR-format file (default: POSCAR)

    Returns:
        (name, lattice_vectors, atom_positions, atom_types) tuple:
            name -- title line read from POSCAR file
            lattice_vectors -- list of 3 NumPy vectors
            atom_types_counts -- (atom_types, atom_counts)
            atom_positions -- list of N_a NumPy vectors

    Notes:
        This function only accepts VASP 5-format POSCAR files in fractional ("direct") coordinates.
    """

    # General Exception for parsing errors.

    err_except = Exception(
        "Unable to parse POSCAR file - this function can only read files in the VASP 5 format."
        )

    name = None
    v_latt = None
    atom_types_counts = None
    atom_pos = None

    with open(file_path, 'r') as input_reader:
        # Read title line.

        name = next(input_reader).strip()

        # Read scale factor.

        scale = float(
            next(input_reader).strip()
            )

        # Read lattice vectors.

        v_latt = [
            np.array([float(val) * scale for val in next(input_reader).strip().split()[:3]], dtype = np.float64)
                for _ in range(0, 3)
            ]

        # Read atom types/counts.

        atom_types = next(input_reader).strip().split()

        atom_counts = [
            int(val) for val in next(input_reader).strip().split()
            ]

        if len(atom_types) != len(atom_counts):
            raise err_except
        
        atom_types_counts = (atom_types, atom_counts)

        # Read coordinate type.

        keyword = next(input_reader).strip()[0]

        if keyword.lower() == 's':
            # Ignore selective dynamics.

            keyword = next(input_reader).strip()[0]

        if keyword.lower() != 'd':
            raise Exception("Error: POSCAR file must be in fractional (\"direct\") coordinates.")

        # Read atom positions.

        atom_pos = [
            np.array([float(val) for val in next(input_reader).strip().split()[:3]], dtype = np.float64)
                for _ in range(0, sum(atom_counts))
            ]

    # Return name, lattice vectors, atom types/counts and atom positions.

    return (name, v_latt, atom_types_counts, atom_pos)

def _ReadPhononModes(file_path = r"OUTCAR"):
    """
    Read an OUTCAR file and return phonon frequencies, eigenvectors and atomic masses.

    Params:
        file_path -- path to an OUTCAR file (default: POSCAR)

    Returns:
        (frequencies, eigenvectors, atomic_masses) tuple:
            frequencies -- list of 3N frequencies
            eigenvectors -- list of 3N N_a x 3 NumPy arrays
            atomic_masses -- list of 3N atomic masses
    """

    # Regular expressions.

    num_ions_regex = re.compile("ions per type =\s*(?P<num_ions>(\d+\s+)+)")
    
    pomass_regex = re.compile("POMASS =\s*(?P<atomic_masses>(\d+\.\d+\s+)+)")

    phonon_freq_regex = re.compile(
        "\d+\s+(?P<im>f(/i)?)\s*=\s*(\d+\.\d+) THz\s*(\d+\.\d+) 2PiTHz\s*(?P<freq_inv_cm>\d+\.\d+) cm-1\s*(\d+\.\d+) meV"
        )

    # General Exception for parsing errors.

    err_except = Exception(
        "Unable to parse OUTCAR file."
        )

    with open(file_path, 'r') as input_reader:
        # Read atom counts.

        atom_counts = None

        for line in input_reader:
            match = num_ions_regex.search(line)

            if match:
                atom_counts = [
                    int(count) for count in match.group('num_ions').strip().split()
                    ]

                break

        if atom_counts is None:
            raise err_except

        # Read atomic masses.

        atomic_masses = None

        for line in input_reader:
            if line.strip() == "Mass of Ions in am":
                match = pomass_regex.search(
                    next(input_reader)
                    )
                
                if match is None:
                    raise err_except
                
                atomic_masses = [
                    float(val) for val in match.group('atomic_masses').strip().split()
                    ]

                break

        if atomic_masses is None or len(atomic_masses) != len(atom_counts):
            raise err_except

        # Convert atom_counts + atomic_masses to a flat list of atomic masses.

        temp = []

        for n, m in zip(atom_counts, atomic_masses):
            temp += [m] * n

        atomic_masses = temp

        # Read frequencies and eigenvectors.

        num_atoms = sum(atom_counts)

        frequencies, eigenvectors = None, None

        # Scan through the file for the start of the phonon calculation output.

        for line in input_reader:
            if line.strip() == "Eigenvectors and eigenvalues of the dynamical matrix":
                break

        # Skip the next three lines.

        for _ in range(0, 3):
            next(input_reader)

        # Read frequencies and eigenvectors for 3N modes.

        frequencies, eigenvectors = [], []

        for i in range(0, 3 * num_atoms):
            # Read frequency.

            match = phonon_freq_regex.search(
                next(input_reader)
                )

            if match is None:
                raise err_except

            freq = float(
                match.group('freq_inv_cm')
                )

            is_imaginary = match.group('im') == "f/i"

            frequencies.append(
                -1.0 * freq if is_imaginary else freq
                )

            # Skip eigenvector header.

            next(input_reader)

            # Read eigenvector.

            eigenvector = [
                [float(val) for val in next(input_reader).strip().split()[3:6]]
                    for _ in range(0, num_atoms)
                ]

            eigenvectors.append(
                np.array(eigenvector, dtype = np.float64)
                )

            # Skip blank line.

            next(input_reader)

    # Return frequencies, eigenvectors and atomic masses.

    return (frequencies, eigenvectors, atomic_masses)


if __name__ == "__main__":
    # Check required VASP input files are present.

    for req_file in r"INCAR", r"POTCAR":
        if not os.path.isfile(req_file):
            print(r"Error: Required file '{0}' not found.".format(req_file))
            sys.exit(1)

    # Check there are no folders with the set JobDirPrefix.
    # (A smarter script should handle this properly.)

    job_dirs = glob.glob(
        "{0}-*".format(JobDirPrefix)
        )

    if len(job_dirs) > 0:
        print("Error: Files and/or folders with prefix '{0}' already exist - please change JobDirPrefix.".format(JobDirPrefix))
        sys.exit(1)

    # Read reference structure.

    name, v_latt, atom_types_counts, atom_pos = _ReadStructure(PhononPOSCAR)

    # Read frequencies, eigenvectors and atomic masses.

    frequencies, eigenvectors, atomic_masses = _ReadPhononModes(PhononOUTCAR)

    # Sanity check.

    if len(atomic_masses) != len(atom_pos):
        raise Exception("Error: Number of atoms in PhononPOSCAR is not consistent with the number of normal modes in PhononOUTCAR.")

    if abs(BandIndex) > len(frequencies):
        raise Exception("Error: BandIndex = {0} is out of range for 3N = {1} phonon modes.".format(BandIndex, len(frequencies)))

    # Setup.

    freq, e_vec = frequencies[BandIndex], eigenvectors[BandIndex]

    # Convert eigenvector components to Cartesian displacements.

    for i, mass in enumerate(atomic_masses):
        e_vec[i, :] /= math.sqrt(mass)

    # Convert atomic positions from fractional -> Cartesian coordinates.

    v_1, v_2, v_3 = v_latt

    atom_pos_cart = [
        f_1 * v_1 + f_2 * v_2 + f_3 * v_3
            for f_1, f_2, f_3 in atom_pos
        ]

    # Incerse of lattice vectors for reverse Cartesian -> fractional transformation.

    frac_trans_mat = np.linalg.inv(v_latt)

    # Prepare output file with headers.

    output_file = r"{0}.dat".format(JobDirPrefix)

    with open(output_file, 'w') as output_writer:
        output_writer.write(
            "{0: >15}  {1: >12}\n".format(r"Q [amu^1/2 \AA]", r"E_0 [eV]")
            )

    # Loop over normal-mode amplitudes.

    q_amp = np.arange(
        QMin, QMax + QInc / 10.0, QInc
        )

    for q in q_amp:
        # Set up a folder for the single-point energy calculation.

        job_dir = "{0}-Q_{1}{2:0>6.3f}".format(JobDirPrefix, '+' if q >= 0.0 else '-', abs(q))

        os.mkdir(job_dir)

        # Copy INCAR/POTCAR and KPOINTS, if required, from the startup directory.

        for input_file in r"INCAR", r"POTCAR":
            shutil.copy(
                input_file, os.path.join(job_dir, input_file)
                )

        if os.path.isfile(r"KPOINTS"):
            shutil.copy(
                r"KPOINTS", os.path.join(job_dir, r"KPOINTS")
                )

        # Generate and write out displaced structure.

        atom_pos_new = [
            np.dot(frac_trans_mat, pos_cart) % 1.0
                for pos_cart in atom_pos_cart + q * e_vec
            ]

        with open(os.path.join(job_dir, r"POSCAR"), 'w') as output_writer:
            # Title line.

            output_writer.write(
                "{0}\n".format(name)
                )

            # Scale factor.

            output_writer.write(
                "  {0: >19.16f}\n".format(1.0)
            )

            # Lattice vectors.

            for ax, ay, az in v_latt:
                output_writer.write("  {0: >21.16f}  {1: >21.16f}  {2: >21.16f}\n".format(ax, ay, az))

            # Atom types/counts.

            atom_types, atom_counts = atom_types_counts

            output_writer.write(
                "".join("  {0: >3}".format(atom_type) for atom_type in atom_types) + '\n'
                )

            output_writer.write(
                "".join("  {0: >3}".format(atom_count) for atom_count in atom_counts) + '\n'
                )

            # Atom positions.

            output_writer.write("Direct\n")

            for f_1, f_2, f_3 in atom_pos_new:
                output_writer.write(
                    "  {0: >21.16f}  {1: >21.16f}  {2: >21.16f}\n".format(f_1, f_2, f_3)
                    )

        # Run VASP.

        os.chdir(job_dir)

        os.system(
            "{0} | tee \"../{1}.out\"".format(RunVASP, job_dir)
            )

        os.chdir("..")

        # Check VASP job completed successfully.

        result = _CheckComplete(
            r"{0}/OUTCAR".format(job_dir)
            )

        if result == -1:
            print("Error: A VASP run failed to complete.")
            sys.exit(1)

        # Extract total energy and write to output file.

        e0 = _ReadE0(
            r"{0}/OSZICAR".format(job_dir)
            )

        with open(output_file, 'a') as output_writer:
            output_writer.write(
                "{0: >15.3f}  {1: >12.6f}\n".format(q, e0)
                )
