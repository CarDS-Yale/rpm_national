# Remote Patient Monitoring (RPM) Trends Analysis Using AHA Data (2018–2022)

This repository contains data preparation, analysis, and visualization code for examining trends in Remote Patient Monitoring (RPM) adoption among U.S. hospitals between 2018 and 2022. The project uses American Hospital Association (AHA) survey data merged with CMS data to explore patterns in RPM availability, disparities by hospital characteristics, and temporal trends.

## Repository Structure

```bash
📁 cohort_creation.py            # Creates derived variables and prepares the analytic dataset  
📁 tables_rpm_national.py        # Generates descriptive statistics and trend analyses  
📁 figures_plots_rpm_national.py # Plots national trends, subgroup differences, and RPM uptake maps  

Key Features
Assigns U.S. regions and derives hospital characteristics (ownership, teaching status, size, etc.)

Creates time-varying indicators of RPM availability by hospital and year

Implements trend analysis via generalized linear and ANCOVA-style models

Visualizes RPM uptake using bar plots, line plots, KDE distributions, and geospatial maps

## Requirements
Python >= 3.8  
pandas  
matplotlib  
seaborn  
statsmodels  
geopandas  

##Usage
Prepare the dataset
Run cohort_creation.py to generate derived variables and construct the analytic cohort.

##Generate summary tables
Execute tables_rpm_national.py to compute national and subgroup statistics.

##Visualize trends
Use figures_plots_rpm_national.py to create publication-quality figures and geospatial maps.

All output figures are saved as .svg or .png in your specified output directory.

##Citation
If you use this code in your research, please cite:
Pedroso AF, et al. National Patterns of Remote Patient Monitoring Service Availability at US Hospitals.