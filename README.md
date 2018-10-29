<p align="center">
  <a href="LICENSE" alt="GPLv3 license"><img src="https://img.shields.io/badge/license-GPLv3-brightgreen.svg" /></a>
  <a href="#" alt="no warranty"><img src="https://img.shields.io/badge/warranty-no-red.svg" /></a>
</p>

# SFRP_Topology_Optimization

A Python tool for the topology optimization of short fiber reinforced plastics.

## Requirements

The following libraries are required:
* [FEniCS](https://fenicsproject.org/)
* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/) incl. numpy headers
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) incl. QWebEngine (QWebKit also works)

If unsure, run the tool and check the error messages of missing packages.

## Run

Enter the following command to run the GUI (with an optional project file to load)
```
python run.py
```
A list of available command line options can be shown with
```
python run.py --help
```

## Acknowledgements

[Felix Ospald](https://www.tu-chemnitz.de/mathematik/part_dgl/people/ospald) gratefully acknowledges financial support by the [German Research Foundation](http://www.dfg.de/en/) (DFG), [Federal Cluster of Excellence EXC 1075](https://www.tu-chemnitz.de/MERGE/) "MERGE Technologies for Multifunctional Lightweight Structures". Many thanks to [Matti Schneider](https://www.itm.kit.edu/cm/287_3957.php) for his helpful introduction to FFT-based homogenization and ideas regarding the ACG distribution.
