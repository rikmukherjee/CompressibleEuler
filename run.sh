#!/bin/bash

for n in $(seq 14 1 14); do
    N=$(echo "2^$n" | bc)  # Calculate 2^n using bc
    nu="5.0d-1"
    delta=$(echo "5.0/5.0" | bc -l)  # Calculate 2/5 using bc
    tFinal="80000"
    # Modify the Fortran file with sed
    sed -i "4s/.*/	integer,parameter::N=${N}/" blast.f90
    sed -i "5s/.*/	real(kind=8),parameter::nu=${nu}/" blast.f90
    sed -i "6s/.*/	real(kind=8),parameter::delta=${delta}/" blast.f90
    sed -i "7s/.*/	real(kind=8),parameter::tFinal=${tFinal}/" blast.f90

    # Create a folder based on n and nu values
    foldername=$(printf "N-%d-nu-%s-delta-%.2f" "$n" "$nu" "$delta")
    # Compile the Fortran code
    gfortran 1D_FreeEvolution.f90 -lfftw3 -L/home/ritwik/Softwares/fftw-3.3.10/lib -lm 

    echo ${foldername}
    mkdir $foldername

    # Move and copy files
    mv a.out ${foldername}/.
    cp blast.f90 ${foldername}

    # Change directory and run the program
    cd ${foldername}
    ./a.out
    cd ..  # Go back to the parent directory
done
