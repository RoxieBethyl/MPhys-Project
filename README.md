**Author:** Blythe Fernandes

# <ins>MPhys Research Project:</ins> Hunting Emission Lines in High Redshift Galaxies with JWST

This repository is a record of the work I undertook while working on my research project during my final year at the University of Edinburgh.

### <ins>Directories</ins>
The purpose of the folders created during this project are listed below:
- [`Filter_files`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Filter_files): Holds the telescope datafiles. Taken from [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse)
- `Images`: Output directory for the images used to present and describe findings in the final report.
- [`imgprocesslib`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/imgprocesslib): Library of the scripts files developed for data analysis.
- `Jades_files` (Not part of this repository): Data images taken from JWST. Analysis was performed on these data files.
- [`Output`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output): Main directory for all output files generated from code run in `imgprocesslib`.
  - [`Output\Catalogue`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output/Catalogue): Master catalogue of relevant objects used for anaylsis is output into this directory. These files are used to produce plots to visualise results.
- [`Photoz`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Photoz): Files necesary to run EaZY Photoz, taken from [eazy-photoz](https://github.com/gbrammer/eazy-photoz/), are stored in this folder.
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
- `Init_Starlink.txt`: Contains the cmd lines to start Starlink on WSL.
- `EazyRun.param`: Holds values of paramters that manage the type of run with [EaZY](https://github.com/gbrammer/eazy-py?tab=readme-ov-file).