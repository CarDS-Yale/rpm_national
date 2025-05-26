#TABLE 1: TableOne for RPM National Data
#Import Packages:
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_chisquare

path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'

#%%
# Load your data
data = pd.read_stata(path + 'aha_18-22_final_dataset.dta')

#%%
def calculate_descriptive_stats(df, group_vars, rpm_vars):
    results = []
    
    # First row: total (not grouped)
    # Add total N row
    total_row = ['Total N']
    for rpm_any in rpm_vars:
        total_count = df[rpm_any].count()
        total_row.append(f"N = {total_count}")
    results.append(total_row)
    
    # Process each categorical variable
    for group in group_vars:
        group_data = df.groupby(group)
        for name, group_df in group_data:
            row = [f'{group}: {name}']
            for rpm in rpm_vars:
                count = int(group_df[rpm].sum())
                percentage = count / group_df[rpm].count() * 100
                row.append(f"{count} ({percentage:.1f}%)")
            results.append(row)
    
    return results

group_vars = ['size', 'us_regions', 'area_type', 'teaching', 'ownership']

rpm_vars = ['rpm_ever', 'rpm_any18', 'rpm_any19', 'rpm_any20', 'rpm_any21', 'rpm_any22']

#%%
# Generate descriptive statistics
descriptive_stats = calculate_descriptive_stats(data, group_vars, rpm_vars)

columns = ['Characteristic'] + [f"{rpm} N(%)" for rpm in rpm_vars]

descriptive_df = pd.DataFrame(descriptive_stats, columns=columns)

# Export to CSV
descriptive_df.to_csv(path + 'descriptive_statistics.csv', index=False)

# Display the table
print(descriptive_df)

#%%
# Export to CSV
descriptive_df.to_csv(path + 'descriptive_table_new2.csv', index=False)

# %%
#Descriptive using err ≤1
def calculate_descriptive_stats(df, group_vars, rpm_vars, errhf_vars, errami_vars):
    results = []
    
    # Add total N row
    total_row = ['Total N']
    for rpm in rpm_vars:
        total_count = df[rpm].count()
        total_row.append(f"N = {total_count}")
    results.append(total_row)
    
    # Process each categorical variable
    for group in group_vars:
        group_data = df.groupby(group)
        for name, group_df in group_data:
            row = [f'{group}: {name}']
            for rpm in rpm_vars:
                count = int(group_df[rpm].sum())
                percentage = count / group_df[rpm].count() * 100
                row.append(f"{count} ({percentage:.1f}%)")
            results.append(row)
    
    return results

group_vars = ['size', 'us_regions', 'area_type', 'teaching', 'ownership']
rpm_vars = ['rpm_ever', 'rpm_any18', 'rpm_any19', 'rpm_any20', 'rpm_any21', 'rpm_any22']

descriptive_stats = calculate_descriptive_stats(data, group_vars, rpm_vars)

columns = ['Characteristic'] + rpm_vars

descriptive_df = pd.DataFrame(descriptive_stats, columns=columns)

# Display the table
print(descriptive_df)
# %%
descriptive_df.to_csv('descriptive_table2.csv', index=False)
# %%
from scipy import stats
from statsmodels.stats.proportion import proportions_chisquare
from scipy.stats import norm  

#%%
# Cochran-Armitage test for trend
def cochran_armitage_test(counts, nobs):
    table = np.array([counts, nobs - counts]).T
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    grand_total = table.sum()
    
    expected = np.outer(row_totals, col_totals) / grand_total
    chi2 = ((table - expected) ** 2 / expected).sum()
    
    weights = np.arange(len(counts))
    weighted_counts = np.dot(counts, weights)
    weighted_totals = np.dot(row_totals, weights)
    
    numerator = weighted_counts - (weighted_totals * col_totals[0] / grand_total)
    denominator = np.sqrt((weighted_totals * col_totals[0] * col_totals[1]) / grand_total / (grand_total - 1))
    
    z = numerator / denominator
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return p_value

def calculate_descriptive_stats_with_trend(df, group_vars, rpm_vars, errhf_vars, errami_vars):
    results = []
    
    # Add total N row
    total_row = ['Total N']
    for rpm in rpm_vars:
        total_count = df[rpm].count()
        total_row.append(f"N = {total_count}")
    results.append(total_row + [np.nan])  # Add an empty value for the p-value column
    
    # Process each categorical variable
    for group in group_vars:
        group_data = df.groupby(group)
        for name, group_df in group_data:
            row = [f'{group}: {name}']
            counts = []
            nobs = []
            for rpm in rpm_vars:
                count = int(group_df[rpm].sum())
                counts.append(count)
                nobs.append(group_df[rpm].count())
                percentage = count / group_df[rpm].count() * 100
                row.append(f"{count} ({percentage:.1f}%)")
            p_value = cochran_armitage_test(np.array(counts), np.array(nobs))
            results.append(row + [p_value])
    
    # Add row for ERRHF
    errhf_row = ['ERRHF']
    counts = []
    nobs = []
    for err_var in errhf_vars:
        count = int(df[err_var].sum())
        counts.append(count)
        nobs.append(df[err_var].count())
        percentage = count / df[err_var].count() * 100
        errhf_row.append(f"{count} ({percentage:.1f}%)")
    p_value = cochran_armitage_test(np.array(counts), np.array(nobs))
    results.append(errhf_row + [p_value])
    
    # Add row for ERRAMI
    errami_row = ['ERRAMI']
    counts = []
    nobs = []
    for err_var in errami_vars:
        count = int(df[err_var].sum())
        counts.append(count)
        nobs.append(df[err_var].count())
        percentage = count / df[err_var].count() * 100
        errami_row.append(f"{count} ({percentage:.1f}%)")
    p_value = cochran_armitage_test(np.array(counts), np.array(nobs))
    results.append(errami_row + [p_value])
    
    return results

group_vars = ['bed_size3', 'us_regions', 'area_type', 'teaching2', 'ownership']
rpm_vars = ['rpm_ever', 'rpm18_any', 'rpm19_any', 'rpm20_any', 'rpm21_any', 'rpm22_any']
errhf_vars = ['errhf_15_18_d', 'errhf_16_19_d', 'errhf_17_20_d', 'errhf_18_21_d', 'errhf_19_22_d']
errami_vars = ['errami_15_18_d', 'errami_16_19_d', 'errami_17_20_d', 'errami_18_21_d', 'errami_19_22_d']

descriptive_stats = calculate_descriptive_stats_with_trend(data, group_vars, rpm_vars, errhf_vars, errami_vars)

columns = ['Characteristic'] + rpm_vars + ['P-value']

descriptive_df = pd.DataFrame(descriptive_stats, columns=columns)

# Export to CSV
#descriptive_df.to_csv(path + 'descriptive_statistics_with_pvalues.csv', index=False)

# Display the table
print(descriptive_df)

# %%
def calculate_descriptive_stats_rpm0(df, group_vars, rpm_vars):
    results = []
    
    # First row: total (not grouped)
    total_row = ['Total N']
    for rpm_any in rpm_vars:
        total_count = df[rpm_any].count()
        rpm0_count = (df[rpm_any] == 0).sum()
        total_row.append(f"N = {rpm0_count}")
    results.append(total_row)
    
    # Process each categorical variable
    for group in group_vars:
        group_data = df.groupby(group)
        for name, group_df in group_data:
            row = [f'{group}: {name}']
            for rpm in rpm_vars:
                count = (group_df[rpm] == 0).sum()
                total = group_df[rpm].count()
                percentage = (count / total * 100) if total > 0 else 0
                row.append(f"{count} ({percentage:.1f}%)")
            results.append(row)
    
    return results

# %%
# Generate descriptive statistics for RPM=0
group_vars = ['size', 'us_regions', 'area_type', 'teaching', 'ownership']

rpm_vars = ['rpm_ever', 'rpm_any18', 'rpm_any19', 'rpm_any20', 'rpm_any21', 'rpm_any22']

descriptive_stats_rpm0 = calculate_descriptive_stats_rpm0(data, group_vars, rpm_vars)

columns = ['Characteristic'] + [f"{rpm} N(%)" for rpm in rpm_vars]

descriptive_df_rpm0 = pd.DataFrame(descriptive_stats_rpm0, columns=columns)

# Export to CSV
descriptive_df_rpm0.to_csv(path + 'descriptive_statistics_rpm0.csv', index=False)


#TABLE 2: TableOne for RPM National Data
import pandas as pd
import numpy as np
from tableone import TableOne, load_dataset

#%%
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'

df = pd.read_stata(path + 'aha_18-22_merged_cities.dta')

#%%
# Define columns, categorical, continuous, groupby, and nonnormal variables
columns = ['age_over_65', 'female', 'race_black', 'hispanic', 'income_household_median', 
           'education_less_highschool', 'health_uninsured', 'disabled'] 
continuous = ['age_over_65', 'race_black', 'hispanic', 'female', 'income_household_median', 'education_less_highschool', 'disabled']
groupby = 'rpm_ever'
nonnormal = ['age_over_65', 'race_black', 'hispanic', 'female', 'income_household_median', 'education_less_highschool', 'disabled']


# Create the TableOne object
mytable = TableOne(df, columns=columns, continuous=continuous, groupby=groupby, nonnormal=nonnormal, pval=True, smd=True,
                  htest_name=True, dip_test=True, normal_test=True, tukey_test=True)

# Print the TableOne summary
print(mytable.tabulate(tablefmt="fancy_grid"))

# %%
#Export table as csv file
# Convert the table to a DataFrame and export as CSV
# Convert the TableOne summary to a DataFrame and export as CSV
table_df = mytable.tableone
table_df.to_csv('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/table_2_new.csv', index=False)
# %%
