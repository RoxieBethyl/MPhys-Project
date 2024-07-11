**Author:** Blythe Fernandes

# <ins>MPhys Research Project:</ins> Hunting Emission Lines in High Redshift Galaxies with JWST

This repository is a record of the work I undertook while working on this project during my final year at the University of Edinburgh.

### <ins>Directories</ins>
The usage of the folders created, during this project are listed below:
- [`Filter_files`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Filter_files): Holds information for the transmission output of the telescope. Taken from [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse)
- `Images`: Output directory for images used to present and describe findings in the final report.
- [`imgprocesslib`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/imgprocesslib): Custom made library of scripts files developed for data analysis.
- `Jades_files` (Not part of repository): Data images taken from JWST. Analysis is performed on these data files.
- [`Output`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output): Main directory for all output files generated from code run in `imgprocesslib`.
  - [`Output\Catalogue`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output/Catalogue): Master catalogue of relevant objects used for anaylsis. Output from `imgprocesslib` stord in this directory. These files are used to constuct plots to visualise results.
- [`Photoz`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Photoz): Files necessary to run EaZY Photoz, taken from [eazy-photoz](https://github.com/gbrammer/eazy-photoz/), are stored in this folder.
- `Trash`: Here, all irrelevant files are dumped - in case for future reference.

### <ins>Notebooks</ins>
The analysis of the research is grouped into each notebook.
- `Aper, Conv, Calib.ipynb`
- `Depth Analysis.ipynb`
- `Filter Maps.ipynb`
- `Getting Results.ipynb` (Linux compatible version: `Getting Results_linux.ipynb`)
- `Make Catalogue.ipynb` (Linux compatible version: `Make Catalogue_linux.ipynb`)
- `MPhys Proj.ipynb`

### <ins>Script Files:</ins>
- `eazy_run.py`: Script that runs EaZY on the input file with parameters from `EazyRun.param`
- `run_exe.py`: Contains custom functions that are called in directly from the file.

### <ins>Files:</ins>
- `Init_Starlink.txt`: Contains command lines to start Starlink on WSL.
- `EazyRun.param`: Holds values of paramters that manage the type of run with [EaZY](https://github.com/gbrammer/eazy-py?tab=readme-ov-file).