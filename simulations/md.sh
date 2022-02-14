#!/bin/sh
#PBS -N _md_fcu_0MPa
#PBS -l walltime=71:59:00
#PBS -l nodes=1:ppn=1
#PBS -m n

date

# Set up input
ORIGDIR=$PBS_O_WORKDIR
WORKDIR=/local/$PBS_JOBID

if [ ! -d $WORKDIR ]; then mkdir -p $WORKDIR; fi
cd $WORKDIR

cp ${ORIGDIR}/npt/md_npt_mof.py $WORKDIR
cp ${ORIGDIR}/npt/ff_lammps.py $WORKDIR
cp ${ORIGDIR}/data/uio66/ff/pars_fcu.txt $WORKDIR
cp ${ORIGDIR}/data/uio66/struct/fcu.chk $WORKDIR

# Copy back results every half hour
( while true; do
	sleep 1800
	cp ${WORKDIR}/output_fcu_0MPa.h5 ${ORIGDIR}/data/uio66/md/
    cp ${WORKDIR}/restart0_output_fcu_0MPa.h5 ${ORIGDIR}/data/uio66/md/
	cp ${WORKDIR}/fcu_0MPa.log ${ORIGDIR}/data/uio66/md/
  done ) &


# Load modules
module load LAMMPS/3Mar2020-foss-2019b-Python-3.7.4-kokkos # Load LAMMPS, also loads yaff

# Run
python md_npt_mof.py > fcu_0MPa.log

# Copy back results
cp ${WORKDIR}/output_fcu_0MPa.h5 ${ORIGDIR}/data/uio66/md/
cp ${WORKDIR}/restart0_output_fcu_0MPa.h5 ${ORIGDIR}/data/uio66/md/
cp ${WORKDIR}/fcu_0MPa.log ${ORIGDIR}/data/uio66/md/

# Finalize
rm -rf $WORKDIR

date
