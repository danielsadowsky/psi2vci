# psi2vci
A python3 script to generate anharmonic potential energy surfaces for polyatomic molecules. The script is in active development and currently has limited functionality. A geometry file is required (see h2o.xyz for an example), as is a Hessian in PSI4 format (see h2o.hess for an example). The molecular graph can be specified by a connectivity file containing a list of covalent bonds (see h2o.conn for an example); if no such file is specified, a guess is made using the geometry and Hessian. For anharmonic bonding potentials, either a total atomization energy must be provided from the command line, or alternatively bond dissociation energies can be provided in the connectivity file. The script generates a .ff file which is readable by PyPES, and after conversion to normal coordinates, can be used to run VCI calculations in PyVCI. The following command, using data from a DF-MP2/cc-PVTZ calculation of the structure of a water molecule, can be used as a test: 

python3 psi2vci.py -V -n 4 -t 0.355240287 -x h2o.xyz -y h2o.hess -c h2o.conn -o h2o.ff

A full list of options can be found in the help menu:

python3 psi2vci.py -h

A description of the available bicoordinate potentials can be found here:

https://doi.org/10.1063/5.0096893

