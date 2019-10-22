# Opt.py


JobDirPrefix = "Opt"

RunVASP = r"mpirun -ppn 36 -n 36 vasp_std"


import glob
import os
import re
import shutil
import sys


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
        print("Error: Folders with prefix '{0}' already exist - please change JobDirPrefix.".format(JobDirPrefix))
        sys.exit(1)
    
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
    
    # Keep track of the run number and current POSCAR file.
    # After each successful run, current_poscar is updated to point at the CONTCAR file.
    
    run_num = 1
    current_poscar = r"POSCAR"
    
    # Run VASP until a run does N < NSW steps.
    
    while True:
        # Set up a folder for the current VASP job.
        
        job_dir = r"{0}-{1:0>3}".format(JobDirPrefix, run_num)
        
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
        
        # Copy current_poscar -> POSCAR.
        
        shutil.copy(
            current_poscar, os.path.join(job_dir, r"POSCAR")
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
        
        # Update current_poscar.
        
        current_poscar = "{0}/CONTCAR".format(job_dir)
        
        # If VASP ran < NSW ionic steps, assume the optimisation is finished and stop.
        
        if result < nsw:
            # If not already present, try and copy the optimised POSCAR to POSCAR.Opt.
            # Otherwise, just print the location of the optimised POSCAR.
            
            if not os.path.isfile(r"POSCAR.Opt"):
                shutil.copy(current_poscar, r"POSCAR.Opt")
                
                print("INFO: Optimised POSCAR copied to POSCAR.Opt.")
                
            else:
                print("INFO: Optimised POSCAR at: {0}".format(current_poscar))
                
            sys.exit(0)
        
        # If not, update run_number and run again.
        
        run_num += 1
