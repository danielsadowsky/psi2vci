# psi2vci
A python3 script to generate anharmonic potential energy surfaces for polyatomic molecules. In its present form, the script only works for symmetric triatomic molecules. A geometry file with connectivity information is required (see h2o.xyz for an example) is required, as is a Hessian in PSI4 format (see h2o.hess for an example), and a total atomization energy (which is supplied through the command line). The script generates a .ff file which is readable by PyPES, and after conversion to normal coordinates, can be used to run VCI calculations in PyVCI. The following command, using data from a DF-MP2/PTZ calculation on a water molecule, can be used as a test: 

python3 psi2vci.py -PV -n 4 -t 0.355240287 -x h2o.xyz -y h2o.hess -o h2o.ff

A full list of options can be found in the help menu:

python3 psi2vci.py -h
