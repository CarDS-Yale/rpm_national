# RPM_National: Evaluating the Availability of Remote Patient Monitoring at U.S. Hospitals

*Aline F. Pedroso PhD, Zhenqiu Lin PhD, Joseph S. Ross MD, MHS Rohan Khera MD, MS*

[[`Manuscript`](https://www.medrxiv.org/content/10.1101/2024.10.14.24315496v1)] [[`CarDS Lab`](https://www.cards-lab.org)]

-----

This repository contains the codebase used in the study titled:

**"National Patterns of Remote Patient Monitoring Service Availability at U.S. Hospitals"**

The study evaluates the presence and temporal trends in Remote Patient Monitoring (RPM) service availability using U.S. hospital-level data, leveraging claims and public datasets to explore RPM adoption across settings, specialties, and patient groups.

---

## Overview

The repository includes all analytical code to generate:

1. **Cohort Construction**: Defines a national hospital cohort using Medicare claims and facility-level datasets.
2. **Descriptive Tables**: Summarizes characteristics of hospitals offering RPM.
3. **Publication Figures**: Visualizes RPM availability and distribution across hospital characteristics and geography.

The repository is intended to promote transparency, reproducibility, and reuse of methodology for hospital-based health services research using administrative data.

---

## File Descriptions

### `cohort_creation.py`

This script assembles a cohort of U.S. hospitals reporting use of RPM services. Key tasks include:

- Identifying hospitals with documented RPM billing codes.
- Merging facility-level characteristics from auxiliary datasets.
- Creating analytic variables for hospital type, region, ownership, and digital infrastructure.
- Outputting a structured analytic dataset for further tabulation and plotting.

---

### `tables_rpm_national.py`

Generates descriptive statistics for the manuscript, including:

- **Table 1**: Baseline characteristics of hospitals with and without RPM services.
- **Table 2**: Stratified analysis of RPM availability by hospital type, ownership, region, and digital maturity.

All tables are exported as CSV files, ready for review and integration into the publication.

---

### `figures_plots_rpm_national.py`

Produces the visualizations in the manuscript, including:

- **Figure 1**: Annual trends in RPM availability.
- **Figure 2**: Geographic map of RPM availability by state.
- **Figure 3**: RPM adoption stratified by rurality and hospital ownership.
- **Figure 4**: Specialty-specific RPM availability across years.
- **Figure 5**: Distribution of RPM services by teaching status and hospital bed size.

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
