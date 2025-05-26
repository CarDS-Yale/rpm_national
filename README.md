# Remote Patient Monitoring Trends (2018–2022)

**Aline Pedroso Camargos**, *et al.*  
[**CarDS Lab, Yale School of Medicine**](https://www.cards-lab.org)

[[`Preprint`](https://www.medrxiv.org/content/10.1101/2024.10.14.24315496v1)] [[`Lab Page`](https://www.cards-lab.org)]

---

<p align="center">
  <img src="content/rpm_summary_figure.png" height="400">
</p>

This repository contains code for processing, analyzing, and visualizing trends in **Remote Patient Monitoring (RPM)** implementation across U.S. hospitals using the AHA Annual Survey data (2018–2022). The analysis explores national patterns and disparities in RPM availability by geography, hospital size, ownership, teaching status, and rurality.

---

## 📁 Repository Structure

```bash
📁 cohort_creation.py            # Creates derived hospital-level features and analytic variables  
📁 tables_rpm_national.py        # Generates descriptive statistics and logistic regression tables  
📁 figures_plots_rpm_national.py # Produces all study figures including maps and trend charts  

---

## ⭐️ Key Features

- Assigns U.S. Census regions based on hospital state  
- Derives hospital ownership, size, rural/urban classification, and teaching status  
- Creates time-varying indicators of RPM availability by hospital and year  
- Runs temporal trend analysis using logistic regression and ANCOVA-style models  
- Produces publication-ready figures: line plots, bar plots, KDE distributions, and geospatial maps  

---

## 🚀 Usage
bash
Copy
Edit

### Step 1: Prepare the dataset
python cohort_creation.py

### Step 2: Generate summary tables
python tables_rpm_national.py

### Step 3: Create plots and maps
python figures_plots_rpm_national.py
Output figures and tables are saved as .svg or .png files in your specified output directory.

## 📊 Example Outputs
<p align="center"> <img src="content/rpm_trend_plot.svg" height="260"> <img src="content/rpm_us_map.svg" height="260"> </p>

## 📄 Citation
yaml
Copy
Edit
Pedroso A, et al. Remote Patient Monitoring Trends in the U.S., 2018–2022. medRxiv. 2025.
bibtex
Copy
Edit


## 📬 Contact
For questions, contact aline.camargos@yale.edu
Website: https://www.cards-lab.org

## 📝 License
MIT License