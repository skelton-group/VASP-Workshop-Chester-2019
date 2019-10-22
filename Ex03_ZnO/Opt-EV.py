# Opt-EV.py


# WARNING:
#  This script is starts each volume by scaling the structure at the initial volume, rather than the structure from the last optimised volume.
#  For complex systems with lots of degrees of freedom, this is at best a waste of time, and at worst may make it more difficult to locate the minimum at each step.


JobDirPrefix = "EV"

RunVASP = r"mpirun -ppn 36 -n 36 vasp_std"

VMin = 0.90
VMax = 1.05
VInc = 0.01


import glob
import os
import re
import shutil
import sys

import numpy as np


def _ReadE0(file_path = r"OSZICAR"):
    """
    Read an OSZICAR file and return the total energy E0 from the last electronic SCF step.

    Params:
      file_path -- path to an OSZICAR-format file (default: OSZICAR)

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

def _CheckComplete(file_path = r"OUTCAR"):
    """
    Inspects an OUTCAR file and (crudely!) determines whether a VASP job has finished cleanly.

    Params:
       file_path -- path to an OUTCAR-format file (default: OUTCAR)

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


if __name__ == "__main__":
    # Check required VASP input files are present.

    for req_file in r"INCAR", r"POSCAR", r"POTCAR":
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

    # Read initial volume from POSCAR and check the positions are specified in fractional ("direct") coordinates.

    initial_volume = None

    with open(r"POSCAR", 'r') as input_reader:
        # Skip title line.

        next(input_reader)

        # Scale factor.

        scale_factor = float(
            next(input_reader).strip()
            )

        # Lattice vectors.

        a_1, a_2, a_3 = [
            [scale_factor * float(value) for value in next(input_reader).strip().split()[:3]]
                for _ in range(0, 3)
            ]

        # Calculate volume (scalar triple product).

        initial_volume = np.dot(
            a_1, np.cross(a_2, a_3)
            )

        # Assume VASP 5 format and skip the atom types/counts lines.

        for _ in range(0, 2):
            next(input_reader)

        keyword = next(input_reader).strip()[0]

        if keyword.lower() == 's':
            # Ignore selective dynamics specification.

            keyword = next(input_reader).strip()[0]

        if keyword.lower() != 'd':
            raise Exception("Error: POSCAR file should be in the VASP 5 format with atom positions in fractional (\"direct\") coordinates.")

    # Read NSW from INCAR.

    nsw = None

    nsw_regex = re.compile(
        r"NSW\s*=\s*(?P<nsw>\d+)"
        )

    with open(r"INCAR", 'r') as input_reader:
        for line in input_reader:
            match = nsw_regex.search(line)

            if match:
                # Specifying the same INCAR tag multiple times is a bad idea...

                if nsw is not None:
                    print("Error: NSW specified multiple times in INCAR.")
                    sys.exit(1)

                nsw = int(
                    match.group('nsw')
                    )

    if nsw is None:
        print("Error: NSW not specified in INCAR.")
        sys.exit(1)

    # Prepare output file with headers.

    output_file = r"{0}.dat".format(JobDirPrefix)

    with open(output_file, 'w') as output_writer:
        output_writer.write(
            "{0: >12}  {1: >12}\n".format(r"V [\AA^3]", r"E_0 [eV]")
            )

    # Loop over volume scale factors and collect optimised total energies.

    v_scale = np.arange(
        VMin, VMax + VInc / 10.0, VInc
        )

    v_opt, e_opt = [], []

    for scale in v_scale:
        # Cell volume.

        v = scale * initial_volume

        # Prefix for job directories.

        job_dir_prefix = "{0}-{1:.3f}".format(JobDirPrefix, scale)

        # For the current scale, keep track of the run number and current POSCAR file.
        # After each successful run, current_poscar is updated to point at the CONTCAR file.

        run_num = 1
        current_poscar = None

        # Run optimisation runs until one does N < NSW steps.

        while True:
            # Set up a folder for the current VASP job.

            job_dir = r"{0}-{1:0>3}".format(job_dir_prefix, run_num)

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

            # If current_poscar is set, copy to POSCAR.
            # If not, write it using the initial POSCAR and the scaled volume.

            if current_poscar is not None:
                shutil.copy(
                    current_poscar, os.path.join(job_dir, r"POSCAR")
                    )
            else:
                with open(r"POSCAR", 'r') as input_reader:
                    with open(os.path.join(job_dir, r"POSCAR"), 'w') as output_writer:
                        # Copy title line.

                        output_writer.write(
                            next(input_reader)
                            )

                        # Replace scale factor in input file with cell volume.

                        next(input_reader)

                        output_writer.write(
                            "-{0:.5f}\n".format(v)
                            )

                        # Copy remainder of POSCAR.

                        for line in input_reader:
                            output_writer.write(line)

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

            # Update current_poscar.

            current_poscar = "{0}/CONTCAR".format(job_dir)

            # If VASP ran < NSW ionic steps, assume the optimisation is finished and stop.

            if result < nsw:
                # Extract total energy and write the volume and energy to the output file.copy the optimised POSCAR to POSCAR-<scale>.Opt.

                e0 = _ReadE0(
                    r"{0}/OSZICAR".format(job_dir)
                    )

                with open(output_file, 'a') as output_writer:
                    output_writer.write(
                        "{0: >12.3f}  {1: >12.6f}\n".format(v, e0)
                        )

                shutil.copy(
                    current_poscar, r"POSCAR-{0:.3f}.Opt".format(scale)
                    )

                break

            # If not, update run_number and run again.

            run_num += 1
