## Instructions for the CSD3 cluster

These instructions are applicable for users with Service Level 3 (free usage) which provides
- Skylake CPUs (Intel Xeon Skylake 6142 processors, 2.6GHz)
- 200,000 CPU core hours per quarter of the year
- Maximum of 448 cores per job
- Maximum of 12h walltime per job
- 6GB of RAM per core (12GB is also possible but might cost more)

### Logging in 

Running the command
```shell
ssh <username>@login-cpu.hpc.cam.ac.uk
```
gets the user into a 'login node' (which has 32 cores and 6GB of RAM per core). Login nodes are usually for preparing code and files for job submission, e.g. compiling code or doing short test runs. They are identical to a single node on the supercomputer.

Longer computing jobs (longer than ~1h or multicore jobs) are not supposed to be run on a login node. Login nodes are being monitored by a watchdog script to prevent this, which can terminate processes.


### File system

There are two available directories.

For non-computing work (e.g. editing reports,...) 
```shell
/home/<username>/
```
is available. This provides a 50GB maximum disk quota and hourly backups but is slower for reading and writing compared to the HPC work directory (see below).

For any HPC input/output, it is recommended to use
```shell
/rds/user/<username>/hpc-work/
```
as this has much faster read/write speeds and a 1TB maximum disk quota (but no backups are made).

In the following, we will assume that the user is in the directory `/rds/user/<username>/hpc-work/`.

### Installing Python

To install Python 3.9.6, run 
```shell
module load python-3.9.6-gcc-5.4.0-sbr552h
```

(This loads a [module](https://docs.hpc.cam.ac.uk/hpc/user-guide/modules.html), specifically a [module provided by Spack](https://docs.hpc.cam.ac.uk/hpc/software-tools/spack.html) (as can be seen by the string `sbr552h`).)

Now, the command `python` calls Python 3.9.6. 

To create a Python virtual environment (`venv`) and upgrade pip, run
```shell
python -m venv venv
source venv/bin/activate
pip install pip setuptools --upgrade
```

Inside the `venv`, all the required packages can then be installed.

Note that, after logging out and logging back in to the login node, the `module` is not loaded anymore so that the command `python` does not refer to the version above. However, the `venv` still points to the correct python version. Therefore, running the `module` command is not necessary anymore, but one can activate the `venv` and access Python from there.

The command `mpirun` is already available without installing any further modules.

#### Installing a different version of Python

All available versions of Python can be shown with the command 
```shell
module avail 2>&1 | grep -i python
```

### Preparing the code

Suppose that 
- the current working directory is `/rds/user/<username>/hpc-work/project/`
- a `venv` is installed as `/rds/user/<username>/hpc-work/venv/`
- the code we would like to run on the supercomputer is `/rds/user/<username>/hpc-work/project/main.py`
- `main.py` can be run with MPI, i.e. as `mpirun -n <ncores> python main.py`

which corresponds to the following directory structure:
```
hpc-work
|- venv
|- project
   |- main.py
   |- (other files which main.py can use)
```

The first thing to do is to create a bash script, say `run.sh`, which activates the virtual environment and runs `main.py`:
#### run.sh
```shell
source /rds/user/<username>/hpc-work/venv/bin/activate
python ./main.py
```

The second thing to do is to create a slurm submission script (SLURM is the workload manager of the cluster). This is a bash script which is submitted to the supercomputer and specifies how the job should be run. Let's call it `hpc_job`. Examples of slurm submission scripts can be found in the directory `/usr/local/Cluster-Docs/SLURM`.

#### hpc_job

The script below will be used to submit a job with 
- Name: `run_1`.
- Project name: `BUSCHER-SL3-CPU` (This should be given by the CSD3 administrators.)
- 14 nodes with 32 cores each, i.e. 448 cores
- 12h of walltime. Note that the time format is hh:mm:ss.
- Path to application executable: `./run.sh`.

All the script does is 
- Configure the correct settings on the supercomputer with the `#SBATCH` directives
- Run the command `mpirun -ppn 32 -np 448 ./run.sh`

```shell
#!/bin/bash
#!
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! Last updated: Mon 13 Nov 12:25:17 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J run_1
#! Which project should be charged:
#SBATCH -A BUSCHER-SL3-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=14
#! How many (MPI) tasks will there be in total? (<= nodes*32)
#! The skylake/skylake-himem nodes have 32 CPUs (cores) each.
#SBATCH --ntasks=448
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! For 6GB per CPU, set "-p skylake"; for 12GB per CPU, set "-p skylake-himem": 
#SBATCH -p skylake

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by core number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*32 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 core by default, and each core is allocated 5980MB (skylake)
#! and 12030MB (skylake-himem). If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MB per node).

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

#! Full path to application executable: 
application="./run.sh"

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 32:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
#CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
```

### Submitting a job

Ensure that the current working directory is where the slurm submission script is located. The directory structure is now
```
project
|- hpc_job
|- run.sh
|- main.py
|- (other files which main.py can use)
```

The slurm submission script is submitted with 
```shell
sbatch hpc_job
```
which returns the job ID, for example `3368222`:
```shell
Submitted batch job 3368222
```

After waiting for the specified walltime, in this case 12 hours, the output created by `main.py` is found in the directory, as well as two additional files created by SLURM:
```
project
|- output of main.py
|- machine.file.3368222
|- slurm-3368222.out
|- hpc_job
|- run.sh
|- main.py
|- (other files which main.py can use)
```

Most importantly, the file `slurm-3368222.out` contains everything that was printed to `stdout` during the run of the job.

### Monitoring jobs

From the time of submission of the submission script, it usually takes a while for the job to start running on the supercomputer (experience says anything from ~2 min to ~2 days). When the job starts running, an email is sent to the user. If the job crashes or terminates successfully, emails are also sent.

To get the scheduled start time of the job, run
```shell
scontrol show job 3368222 | grep Start
```

While a job is running, one can already view the contents of the files `machine.file.3368222` and `slurm-3368222.out`.

A job can be cancelled while it is running by using the command
```shell
scancel 3368222
```

A list of all submitted jobs for the project name `BUSCHER-SL3-CPU` for the user `<username>` and in a specified time interval is printed with
```shell
gstatement -p BUSCHER-SL3-CPU -u <username> -s "2022-09-01-00:00:00" -e "2022-09-19-23:59:59"
```

The current account balance can be viewed with 
```shell
mybalance
```

### Transferring files

Files can be transferred with `rsync`. For example, to copy the directory `project` to the user's machine:
```shell
rsync -av <username>@login-cpu.hpc.cam.ac.uk:/rds/user/<username>/hpc-work/project /Users/<user>
```

To update the contents of a directory which already exists on the user's machine:
```shell
rsync -av <username>@login-cpu.hpc.cam.ac.uk:/rds/user/<username>/hpc-work/project/ /Users/<user>/project/
```

### Known supercomputer issues

If a job crashes, the usual practice of looking at the output in the `slurm-*.out` file and googling the issue applies. 

Known issues when running the code on the supercomputer are:
- Job appears to be running, but the last line in the `slurm-*.out` file is `Socket timed out on send/recv operation`. 
  - This has been confirmed by the CSD3 administrators to be a random hiccup in the supercomputer or network connection.
- Crash with the error message `insufficient virtual memory`. This means that the RAM was exhausted, most likely due to PolyChord allocating too much memory for the live points. Fixes are:
  - Easiest fix: Re-run again without any changes. 
  - Allocate 12GB of RAM to each core by changing the line `#SBATCH -p skylake` to `#SBATCH -p skylake-himem` in the slurm submission script. 
  - Instead of installing the `master` branch of PolyChord, install the `dense_clustering` branch, which uses less memory: `pip install git+https://github.com/PolyChord/PolyChordLite@dense_clustering`. However, at the time of writing, the `dense_clustering` branch is 2 commits behind the `master` branch.
- Python package issues: A Python package which works on the user's local machine throws an error on the supercomputer. (At the time of writing, this has happened with the `anesthetic` and the `joblib` libraries.)
  - Check whether the version on the local machine is the same as the one on the supercomputer. If the versions are the same, check whether the same `git commit` was installed. If the same `commit` is installed, try to replicate the code in the package line by line in the Python interpreter to see where the error occurs.
- The `slurm-*.out` file grows in size until there is no free disk space.
  - Try removing or commenting out any print statements in the code which caused this issue. Note that the output of PolyChord cannot be removed, in which case a different solution must be sought.
- Crash with the error message `double free or corruption`, meaning that some code tried to free some memory twice, causing a segfault. 
  - This error has not been reproducible consistently and occurred extremely rarely and at random in early versions of the code. It has not been found to occur in the newest version of the code. Re-running has fixed the problem every time it occurred.

## More information 
For the full documentation of the CSD3 cluster, see https://docs.hpc.cam.ac.uk/hpc/index.html