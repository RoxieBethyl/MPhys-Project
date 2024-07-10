**Author:** Blythe Fernandes

# MPhys Research Project
## Hunting Emission Lines in High Redshift Galaxies with JWST

The repository is a record of the work I undertook while working on my research project during my final year at the University of Edinburgh.

### <ins>Directories:</ins>
- [`Filter_files`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Filter_files): Holds the telescope datafiles. Taken from [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse)
- `Images`: Output directory for the images used to present and descibe findings in the final report.
- [`imgprocesslib`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/imgprocesslib): Library of the scripts files developed for data analysis.
- `Jades_files` (Not part of this reposistory): Data images taken from JWST. Analysis was performed on these datafiles.
- [`Output`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output): Main directory for all output files generated from code run in `imgprocesslib`.
  - [`Output\Catalogue`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Output/Catalogue): Master catalogue of relevant objects used for anaylsis is output into this directory. These files are used to produces plots to visualise results.
- [`Photoz`](https://github.com/RoxieBethyl/MPhys-Project/tree/main/Photoz): Files necesary to run EaZY Photoz, taken from [`eazy-photoz` repo](https://github.com/gbrammer/eazy-photoz/), are stored in this folder.
- `Trash`: Here, all irrelevant files are dumped - in case for future reference.

### <ins>Notebooks</ins>
The notebooks used for this project are:
- `Aper, Conv, Calib.ipynb`
- `Depth Analysis.ipynb`
- `Filter Maps.ipynb`
- `Getting Results.ipynb` (Linux compatiable version: `Getting Results_linux.ipynb`)
- `Make Catalogue.ipynb` (Linux compatiable version: `Make Catalogue_linux.ipynb`)
- `MPhys Proj.ipynb`

### Script Files:
- `eazy_run.py`
- `run_exe.py`

### <ins>Files:</ins>
- `Init_Starlink.txt`: Contains the cmd lines to startup Starlink on WSL.
- `EazyRun.param`: