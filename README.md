# RPM_National: Evaluating the Availability of Remote Patient Monitoring at U.S. Hospitals

*Aline F. Pedroso, Zhenqiu Lin, Joseph S. Ross, Rohan Khera*

[[`Manuscript`](https://www.medrxiv.org/content/10.1101/2024.10.14.24315496v1)] [[`CarDS Lab`](https://www.cards-lab.org)]

-----

This repository contains the codebase used in the study titled:

**"National Patterns of Remote Patient Monitoring Service Availability at U.S. Hospitals"**

The study evaluates the presence and temporal trends in Remote Patient Monitoring (RPM) service availability using U.S. hospital-level data, leveraging claims and public datasets to explore RPM adoption across settings, specialties, and patient groups.

---

## Overview

The repository includes all analytical code to generate:

1. **Cohort Construction**: Defines a national hospital cohort using the American Hospital Association annual survey data.
2. **Descriptive Tables**: Summarizes characteristics of hospitals offering RPM and the communities served by these hospitals based on county-level census data.
3. **Publication Figures**: Visualizes RPM availability and relative increase in availability across hospital/community characteristics and geography.

The repository is intended to promote transparency, reproducibility, and reuse of methodology for hospital-based health services research using administrative data.

---

## File Descriptions

### `cohort_creation.py`

This script assembles a cohort of U.S. hospitals reporting use of RPM services. Key tasks include:

- Identifying hospitals with documented RPM availability.
- Merging facility-level characteristics from auxiliary datasets.
- Creating analytic variables for hospital type, region, ownership, and digital infrastructure.
- Outputting a structured analytic dataset for further tabulation and plotting.

---

### `tables_rpm_national.py`

Generates descriptive statistics for the manuscript, including:

- **Table 1**: Characteristics of hospitals with RPM services according to size, region, area of location, teaching status and ownership.
- **Table 2**: Baseline characteristics of the communities served by these hospitals.

All tables are exported as CSV files, ready for review and integration into the publication.

---

### `figures_plots_rpm_national.py`

Produces the visualizations in the manuscript, including:

- **Figure 1**: Relative increase in RPM availability from 2018 to 2022.
- **Figure 2**: Geographic map of RPM availability by county.
- **Figure 3**: Community and hospital characteristics associated with the availability of RPM services.
- **Figure 4**: Trends in the proportion of hospitalizations for HF and AMI at hospitals with and without RPM.
- **Figure 5**: Trends in RPM service availability by rural-urban commuting area (RUCA) classification.
.

Figures are generated using `matplotlib`, `seaborn`, and `geopandas`, and saved as high-resolution PNGs.

---

## Requirements & Setup

### Environment

The analysis requires Python 3.8+ and the following Python packages:

```
pandas
matplotlib
seaborn
geopandas
```

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/rpm_national.git
   cd rpm_national
   ```

2. (Optional) Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Adjust input paths and filenames as needed within each script to match your data environment.

---

## Running the Analysis

Run each script sequentially to reproduce the study outputs:

```bash
# Step 1: Generate cohort
python cohort_creation.py

# Step 2: Create summary tables
python tables_rpm_national.py

# Step 3: Generate figures
python figures_plots_rpm_national.py
```

---

## Citation

If you use or adapt this code, please cite the accompanying manuscript:

MLA:
```
Pedroso, A.F., Lin, Z., Ross, J.S., Khera, R. "National Patterns of Remote Patient Monitoring Service Availability at U.S. Hospitals." Medrxiv. 2025.
```

BibTeX:
```bibtex
@article{pedroso2025rpm,
  title={National Patterns of Remote Patient Monitoring Service Availability at US Hospitals},
  author={Pedroso, Aline F. and Lin, Zhenqiu and Ross, Joseph S. and Khera, Rohan},
  journal={Medrxiv},
  year={2025},
  note={Original Research Manuscript}
}
```

---

## Contact

For questions about the dataset or analysis, please contact:

**Aline F. Pedroso, PhD**  
📧 aline.pedroso@yale.edu

**Rohan Khera, MD, MS**  
📧 rohan.khera@yale.edu
