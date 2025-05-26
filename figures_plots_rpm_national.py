# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_18-22_final_dataset.dta')

# %%
# Define the correct order of categories
category_order = ['Hospital Bedsize', 'Region', 'Area Type', 'Teaching Status', 'Ownership']

# Mappings for relabeling
mappings = {
    'size': {1: '<100 beds', 2: '100-300 beds', 3: '>300 beds'},
    'area_type': {'Metro': 'Metropolitan', 'Micro': 'Micropolitan', 'Rural': 'Rural'},
    'teaching': {2: 'Teaching', 1: 'Non-teaching'},
    'ownership': {1: 'Government', 2: 'Private'}
}

# Copy data and apply mappings
df_mapped = df.copy()
for col, mapping in mappings.items():
    df_mapped[col] = df[col].map(mapping)

# Use us_regions directly (already strings)
df_mapped['us_regions'] = pd.Categorical(
    df['us_regions'],
    categories=['Northeast', 'Midwest', 'South', 'West'],
    ordered=True
)

# Set correct ordering for each category
df_mapped['ownership'] = pd.Categorical(df_mapped['ownership'], ['Government', 'Private'], ordered=True)
df_mapped['teaching'] = pd.Categorical(df_mapped['teaching'], ['Teaching', 'Non-teaching'], ordered=True)
df_mapped['area_type'] = pd.Categorical(df_mapped['area_type'], ['Metropolitan', 'Micropolitan', 'Rural'], ordered=True)
df_mapped['size'] = pd.Categorical(df_mapped['size'], ['<100 beds', '100-300 beds', '>300 beds'], ordered=True)

# %%
# Calculate relative change in RPM from 2018 to 2022
results = []

# Loop over mapped categories
for category, subcategories in mappings.items():
    for _, label in subcategories.items():
        group = df_mapped[df_mapped[category] == label]
        rpm18 = group['rpm_any18'].mean()
        rpm22 = group['rpm_any22'].mean()
        if rpm18 and not pd.isna(rpm18):
            rel_change = ((rpm22 - rpm18) / rpm18) * 100
            results.append((label, rel_change, category))

# Add us_regions manually
for region in ['Northeast', 'Midwest', 'South', 'West']:
    group = df_mapped[df_mapped['us_regions'] == region]
    rpm18 = group['rpm_any18'].mean()
    rpm22 = group['rpm_any22'].mean()
    if rpm18 and not pd.isna(rpm18):
        rel_change = ((rpm22 - rpm18) / rpm18) * 100
        results.append((region, rel_change, 'us_regions'))

# %%
# Build the results DataFrame
results_df = pd.DataFrame(results, columns=['Subcategory', 'Difference', 'Category'])

# Rename category labels for plotting
category_names = {
    'ownership': 'Ownership',
    'teaching': 'Teaching Status',
    'us_regions': 'Region',
    'size': 'Hospital Bedsize',
    'area_type': 'Area Type'
}
results_df['Category'] = results_df['Category'].map(category_names)

# Define desired subcategory order for each category
subcategories_order = {
    'Hospital Bedsize': ['<100 beds', '100-300 beds', '>300 beds'],
    'Region': ['Northeast', 'Midwest', 'South', 'West'],
    'Area Type': ['Metropolitan', 'Micropolitan', 'Rural'],
    'Teaching Status': ['Teaching', 'Non-teaching'],
    'Ownership': ['Government', 'Private']
}

# Apply categorical ordering to Subcategory per Category
ordered_subcategories = []
for category in category_order:
    cat_df = results_df[results_df['Category'] == category].copy()
    cat_df['Subcategory'] = pd.Categorical(
        cat_df['Subcategory'],
        categories=subcategories_order[category],
        ordered=True
    )
    ordered_subcategories.append(cat_df)

results_df = pd.concat(ordered_subcategories, ignore_index=True)

# %%
# Reapply category ordering using categorical
results_df['Category'] = pd.Categorical(results_df['Category'], categories=category_order, ordered=True)

# Sort by Category and SortOrder (which is subcategory position within group)
results_df.sort_values(by=['Category', 'SortOrder'], inplace=True)

# Assign plotting positions (top to bottom with space between category blocks)
position_map = {}
pos = 0
for category in category_order:
    subcats = results_df[results_df['Category'] == category]
    for subcat in subcats['Subcategory']:
        position_map[(category, subcat)] = pos
        pos += 1
    pos += 1  # add spacer between category blocks

results_df['Position'] = results_df.apply(lambda row: -position_map[(row['Category'], row['Subcategory'])], axis=1)


# %%
# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

bars = plt.barh(results_df['Position'], results_df['Difference'], color='skyblue')

# Add category headers
for category in category_order:
    sub = results_df[results_df['Category'] == category]
    if sub.empty:
        continue
    y_pos = (sub['Position'].iloc[0] + sub['Position'].iloc[-1]) / 2
    plt.text(-25, y_pos, category, ha='right', va='center', fontsize=12, fontweight='bold')

# Add labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=12)

# Format plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 10))
ax.set_yticks(results_df['Position'])
ax.set_yticklabels(results_df['Subcategory'])
plt.xlabel('Relative Increase in RPM Availability (%)')
plt.tight_layout()
plt.show()

# %%
# Export
fig.savefig(path + 'relative_rpm_change_18_22.png', format='png')
fig.savefig(path + 'relative_rpm_change_18_22.svg', format='svg')

# %%
#FOR POST DISCHARGE
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_18-22_final_dataset.dta')

# %%
# Define the correct order of categories
category_order = ['Hospital Bedsize', 'Region', 'Area Type', 'Teaching Status', 'Ownership']

# Mappings for relabeling
mappings = {
    'size': {1: '<100 beds', 2: '100-300 beds', 3: '>300 beds'},
    'area_type': {'Metro': 'Metropolitan', 'Micro': 'Micropolitan', 'Rural': 'Rural'},
    'teaching': {2: 'Teaching', 1: 'Non-teaching'},
    'ownership': {1: 'Government', 2: 'Private'}
}

# Copy data and apply mappings
df_mapped = df.copy()
for col, mapping in mappings.items():
    df_mapped[col] = df[col].map(mapping)

# Use us_regions directly (already strings)
df_mapped['us_regions'] = pd.Categorical(
    df['us_regions'],
    categories=['Northeast', 'Midwest', 'South', 'West'],
    ordered=True
)

# Set correct ordering for each category
df_mapped['ownership'] = pd.Categorical(df_mapped['ownership'], ['Government', 'Private'], ordered=True)
df_mapped['teaching'] = pd.Categorical(df_mapped['teaching'], ['Teaching', 'Non-teaching'], ordered=True)
df_mapped['area_type'] = pd.Categorical(df_mapped['area_type'], ['Metropolitan', 'Micropolitan', 'Rural'], ordered=True)
df_mapped['size'] = pd.Categorical(df_mapped['size'], ['<100 beds', '100-300 beds', '>300 beds'], ordered=True)

# %%
# Calculate relative change in RPM from 2018 to 2022
results = []

# Loop over mapped categories
for category, subcategories in mappings.items():
    for _, label in subcategories.items():
        group = df_mapped[df_mapped[category] == label]
        rpm18 = group['rpm_pdis18'].mean()
        rpm22 = group['rpm_pdis22'].mean()
        if rpm18 and not pd.isna(rpm18):
            rel_change = ((rpm22 - rpm18) / rpm18) * 100
            results.append((label, rel_change, category))

# Add us_regions manually
for region in ['Northeast', 'Midwest', 'South', 'West']:
    group = df_mapped[df_mapped['us_regions'] == region]
    rpm18 = group['rpm_pdis18'].mean()
    rpm22 = group['rpm_pdis22'].mean()
    if rpm18 and not pd.isna(rpm18):
        rel_change = ((rpm22 - rpm18) / rpm18) * 100
        results.append((region, rel_change, 'us_regions'))

# %%
# Build the results DataFrame
results_df = pd.DataFrame(results, columns=['Subcategory', 'Difference', 'Category'])

# Rename category labels for plotting
category_names = {
    'ownership': 'Ownership',
    'teaching': 'Teaching Status',
    'us_regions': 'Region',
    'size': 'Hospital Bedsize',
    'area_type': 'Area Type'
}
results_df['Category'] = results_df['Category'].map(category_names)

# Define desired subcategory order for each category
subcategories_order = {
    'Hospital Bedsize': ['<100 beds', '100-300 beds', '>300 beds'],
    'Region': ['Northeast', 'Midwest', 'South', 'West'],
    'Area Type': ['Metropolitan', 'Micropolitan', 'Rural'],
    'Teaching Status': ['Teaching', 'Non-teaching'],
    'Ownership': ['Government', 'Private']
}

# Apply categorical ordering to Subcategory per Category
ordered_subcategories = []
for category in category_order:
    cat_df = results_df[results_df['Category'] == category].copy()
    cat_df['Subcategory'] = pd.Categorical(
        cat_df['Subcategory'],
        categories=subcategories_order[category],
        ordered=True
    )
    ordered_subcategories.append(cat_df)

results_df = pd.concat(ordered_subcategories, ignore_index=True)

# %%
# Reapply category ordering using categorical
results_df['Category'] = pd.Categorical(results_df['Category'], categories=category_order, ordered=True)

# Sort by Category and SortOrder (which is subcategory position within group)
results_df.sort_values(by=['Category', 'Subcategory'], inplace=True)


# Assign plotting positions (top to bottom with space between category blocks)
position_map = {}
pos = 0
for category in category_order:
    subcats = results_df[results_df['Category'] == category]
    for subcat in subcats['Subcategory']:
        position_map[(category, subcat)] = pos
        pos += 1
    pos += 1  # add spacer between category blocks

results_df['Position'] = results_df.apply(lambda row: -position_map[(row['Category'], row['Subcategory'])], axis=1)


# %%
# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

bars = plt.barh(results_df['Position'], results_df['Difference'], color='skyblue')

# Add category headers
for category in category_order:
    sub = results_df[results_df['Category'] == category]
    if sub.empty:
        continue
    y_pos = (sub['Position'].iloc[0] + sub['Position'].iloc[-1]) / 2
    plt.text(-25, y_pos, category, ha='right', va='center', fontsize=12, fontweight='bold')

# Add labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=12)

# Format plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 10))
ax.set_yticks(results_df['Position'])
ax.set_yticklabels(results_df['Subcategory'])
plt.xlabel('Relative Increase in RPM Availability (%)')
plt.tight_layout()
plt.show()

# %%
# Export
fig.savefig(path + 'relative_rpmPDIS_change_18_22.png', format='png')
fig.savefig(path + 'relative_rpmPDIS_change_18_22.svg', format='svg')

# %%
#FOR CHRONIC CARE
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_18-22_final_dataset.dta')

# %%
# Define the correct order of categories
category_order = ['Hospital Bedsize', 'Region', 'Area Type', 'Teaching Status', 'Ownership']

# Mappings for relabeling
mappings = {
    'size': {1: '<100 beds', 2: '100-300 beds', 3: '>300 beds'},
    'area_type': {'Metro': 'Metropolitan', 'Micro': 'Micropolitan', 'Rural': 'Rural'},
    'teaching': {2: 'Teaching', 1: 'Non-teaching'},
    'ownership': {1: 'Government', 2: 'Private'}
}

# Copy data and apply mappings
df_mapped = df.copy()
for col, mapping in mappings.items():
    df_mapped[col] = df[col].map(mapping)

# Use us_regions directly (already strings)
df_mapped['us_regions'] = pd.Categorical(
    df['us_regions'],
    categories=['Northeast', 'Midwest', 'South', 'West'],
    ordered=True
)

# Set correct ordering for each category
df_mapped['ownership'] = pd.Categorical(df_mapped['ownership'], ['Government', 'Private'], ordered=True)
df_mapped['teaching'] = pd.Categorical(df_mapped['teaching'], ['Teaching', 'Non-teaching'], ordered=True)
df_mapped['area_type'] = pd.Categorical(df_mapped['area_type'], ['Metropolitan', 'Micropolitan', 'Rural'], ordered=True)
df_mapped['size'] = pd.Categorical(df_mapped['size'], ['<100 beds', '100-300 beds', '>300 beds'], ordered=True)

# %%
# Calculate relative change in RPM from 2018 to 2022
results = []

# Loop over mapped categories
for category, subcategories in mappings.items():
    for _, label in subcategories.items():
        group = df_mapped[df_mapped[category] == label]
        rpm18 = group['rpm_chro18'].mean()
        rpm22 = group['rpm_chro22'].mean()
        if rpm18 and not pd.isna(rpm18):
            rel_change = ((rpm22 - rpm18) / rpm18) * 100
            results.append((label, rel_change, category))

# Add us_regions manually
for region in ['Northeast', 'Midwest', 'South', 'West']:
    group = df_mapped[df_mapped['us_regions'] == region]
    rpm18 = group['rpm_chro18'].mean()
    rpm22 = group['rpm_chro22'].mean()
    if rpm18 and not pd.isna(rpm18):
        rel_change = ((rpm22 - rpm18) / rpm18) * 100
        results.append((region, rel_change, 'us_regions'))

# %%
# Build the results DataFrame
results_df = pd.DataFrame(results, columns=['Subcategory', 'Difference', 'Category'])

# Rename category labels for plotting
category_names = {
    'ownership': 'Ownership',
    'teaching': 'Teaching Status',
    'us_regions': 'Region',
    'size': 'Hospital Bedsize',
    'area_type': 'Area Type'
}
results_df['Category'] = results_df['Category'].map(category_names)

# Define desired subcategory order for each category
subcategories_order = {
    'Hospital Bedsize': ['<100 beds', '100-300 beds', '>300 beds'],
    'Region': ['Northeast', 'Midwest', 'South', 'West'],
    'Area Type': ['Metropolitan', 'Micropolitan', 'Rural'],
    'Teaching Status': ['Teaching', 'Non-teaching'],
    'Ownership': ['Government', 'Private']
}

# Apply categorical ordering to Subcategory per Category
ordered_subcategories = []
for category in category_order:
    cat_df = results_df[results_df['Category'] == category].copy()
    cat_df['Subcategory'] = pd.Categorical(
        cat_df['Subcategory'],
        categories=subcategories_order[category],
        ordered=True
    )
    ordered_subcategories.append(cat_df)

results_df = pd.concat(ordered_subcategories, ignore_index=True)

# %%
# Reapply category ordering using categorical
results_df['Category'] = pd.Categorical(results_df['Category'], categories=category_order, ordered=True)

# Sort by Category and SortOrder (which is subcategory position within group)
results_df.sort_values(by=['Category', 'Subcategory'], inplace=True)


# Assign plotting positions (top to bottom with space between category blocks)
position_map = {}
pos = 0
for category in category_order:
    subcats = results_df[results_df['Category'] == category]
    for subcat in subcats['Subcategory']:
        position_map[(category, subcat)] = pos
        pos += 1
    pos += 1  # add spacer between category blocks

results_df['Position'] = results_df.apply(lambda row: -position_map[(row['Category'], row['Subcategory'])], axis=1)


# %%
# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

bars = plt.barh(results_df['Position'], results_df['Difference'], color='skyblue')

# Add category headers
for category in category_order:
    sub = results_df[results_df['Category'] == category]
    if sub.empty:
        continue
    y_pos = (sub['Position'].iloc[0] + sub['Position'].iloc[-1]) / 2
    plt.text(-25, y_pos, category, ha='right', va='center', fontsize=12, fontweight='bold')

# Add labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=12)

# Format plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 10))
ax.set_yticks(results_df['Position'])
ax.set_yticklabels(results_df['Subcategory'])
plt.xlabel('Relative Increase in RPM Availability (%)')
plt.tight_layout()
plt.show()

# %%
# Export
fig.savefig(path + 'relative_rpmCHRO_change_18_22.png', format='png')
fig.savefig(path + 'relative_rpmCHRO_change_18_22.svg', format='svg')

#%%
#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer

#%%
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/Archive/'

#%%
# Load your data into a pandas DataFrame
data = pd.read_stata(path + 'aha_18-22_merged_cities.dta')

#%%


#%%
# Define your dependent and independent variables
dependent_variable = 'rpm_any22'
independent_variables = [
    'age_over_65', 'female', 'race_black', 'hispanic', 'income_household_median',
    'education_less_highschool', 'disabled', 'size', 'us_regions2', 'area_type', 'teaching', 'ownership'
    ]

# Specify which variables are categorical
categorical_vars = ['size', 'teaching', 'ownership']

# Convert categorical variables into dummies
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# List of all non-categorical variables to be standardized
continuous_vars = ['age_over_65', 'female', 'race_black', 'hispanic', 'education_less_highschool', 'income_household_median', 'disabled'] 

# Standardize the continuous independent variables
scaler = StandardScaler()
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

#%%
#FACTORS ASSOCIATED WITH RPM AVAILABILITY
# Define the formula for the mixed effects logistic regression model
# Note: Ensure variable names from dummies are correctly used here
formula = f"""
{dependent_variable} ~ age_over_65 + female + race_black + hispanic + education_less_highschool + income_household_median + disabled + size_2 + size_3 + C(us_regions2) + C(area_type) + teaching_2 + ownership_2 + (1 | facilityid)
"""

# Fit the mixed effects logistic regression model using pymer4
model = Lmer(formula, data=data, family='binomial')
results = model.fit()

# Print the results summary
results

#%%
# Export the DataFrame to a CSV file
results.to_csv('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/new_mlr.csv', index=False)

# %%
#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer

#%%
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'

#%%
# Load your data into a pandas DataFrame
data = pd.read_stata(path + 'aha_18-22_merged_cities.dta')

#%%


#%%
#FOR POST-DISCHARGE
# Define your dependent and independent variables
dependent_variable = 'rpm_pdis22'
independent_variables = [
    'age_over_65', 'female', 'race_black', 'hispanic', 'income_household_median',
    'education_less_highschool', 'disabled', 'size', 'us_regions2', 'area_type', 'teaching', 'ownership'
    ]

# Specify which variables are categorical
categorical_vars = ['size', 'teaching', 'ownership']

# Convert categorical variables into dummies
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# List of all non-categorical variables to be standardized
continuous_vars = ['age_over_65', 'female', 'race_black', 'hispanic', 'education_less_highschool', 'income_household_median', 'disabled'] 

# Standardize the continuous independent variables
scaler = StandardScaler()
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

#%%

# Define the formula for the mixed effects logistic regression model
# Note: Ensure variable names from dummies are correctly used here
formula = f"""
{dependent_variable} ~ age_over_65 + female + race_black + hispanic + education_less_highschool + income_household_median + disabled + size_2 + size_3 + C(us_regions2) + C(area_type) + teaching_2 + ownership_2 + (1 | facilityid)
"""

# Fit the mixed effects logistic regression model using pymer4
model = Lmer(formula, data=data, family='binomial')
results = model.fit()

# Print the results summary
results

#%%
# Export the DataFrame to a CSV file
results.to_csv('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/new_mlr_pdis.csv', index=False)

# %%
#FOR CHRONIC
# Define your dependent and independent variables
dependent_variable = 'rpm_chro22'
independent_variables = [
    'age_over_65', 'female', 'race_black', 'hispanic', 'income_household_median',
    'education_less_highschool', 'disabled', 'size', 'us_regions2', 'area_type', 'teaching', 'ownership'
    ]

# Specify which variables are categorical
categorical_vars = ['size', 'teaching', 'ownership']

# Convert categorical variables into dummies
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# List of all non-categorical variables to be standardized
continuous_vars = ['age_over_65', 'female', 'race_black', 'hispanic', 'education_less_highschool', 'income_household_median', 'disabled'] 

# Standardize the continuous independent variables
scaler = StandardScaler()
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

#%%

# Define the formula for the mixed effects logistic regression model
# Note: Ensure variable names from dummies are correctly used here
formula = f"""
{dependent_variable} ~ age_over_65 + female + race_black + hispanic + education_less_highschool + income_household_median + disabled + size_2 + size_3 + C(us_regions2) + C(area_type) + teaching_2 + ownership_2 + (1 | facilityid)
"""

# Fit the mixed effects logistic regression model using pymer4
model = Lmer(formula, data=data, family='binomial')
results = model.fit()

# Print the results summary
results

#%%
# Export the DataFrame to a CSV file
results.to_csv('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/new_mlr_chro.csv', index=False)



#%%
#FOREST PLOT FACTORS ASSOCIATED WITH RPM AVAILABILITY
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
# Load the CSV file
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/'
data = pd.read_csv(path + 'new_mlr.csv')

#%%
# Define the desired order of variables
# Define the desired order of variables based on the file
desired_order = [
    'us_regions_4TRUE', 'us_regions_3TRUE', 'us_regions_2TRUE', 'area_type_1.0TRUE', 'area_type_2.0TRUE',
    'ownership_2TRUE', 'teaching_1.0TRUE', 'size_2TRUE', 'size_1TRUE',
    'disabled', 'income_household_median', 'education_less_highschool',
    'hispanic', 'race_black', 'female', 'age_over_65'
]

# Reorder the dataframe based on the desired order
data = data.set_index('Unnamed: 0').loc[desired_order].reset_index()

# Prepare data for the forest plot
variables = data['Unnamed: 0']
estimates = data['OR']
ci_lower = data['OR_2.5_ci']
ci_upper = data['OR_97.5_ci']

# Create the forest plot
fig, ax = plt.subplots(figsize=(8, len(variables) * 0.5))

# Plot the estimates and confidence intervals
ax.errorbar(estimates, np.arange(len(estimates)), xerr=[estimates - ci_lower, ci_upper - estimates], fmt='o', color='black', ecolor='gray', capsize=3)
ax.axvline(x=0, linestyle='--', color='red')

# Set y-axis with variable names
ax.set_yticks(np.arange(len(variables)))
ax.set_yticklabels(variables)

# Set labels
ax.set_xlabel('Odds Ratio')


# Show plot
plt.tight_layout()
plt.show()

# %%
# Define the desired order of variables based on the file, excluding the intercept and inverted
desired_order = [
    'ownership_2TRUE', 'teaching_1.0TRUE', 'area_type_2.0TRUE', 'area_type_1.0TRUE', 'us_regions_2TRUE', 'us_regions_3TRUE', 'us_regions_4TRUE', 'size_1TRUE', 'size_2TRUE',
    'disabled', 'education_less_highschool', 'income_household_median',
    'hispanic', 'race_black', 'female', 'age_over_65'
]

# Reorder the dataframe based on the desired order
data = data.set_index('Unnamed: 0').loc[desired_order].reset_index()

# Prepare data for the forest plot
variables = data['Unnamed: 0']
estimates = data['OR']
ci_lower = data['OR_2.5_ci']
ci_upper = data['OR_97.5_ci']

# Map the new labels to the variables
label_mapping_updated = {
    'us_regions_4TRUE': 'Region, West vs. Northeast',
    'us_regions_3TRUE': 'Region, South vs. Northeast',
    'us_regions_2TRUE': 'Region, Midwest vs. Northeast',
    'area_type_1.0TRUE': 'Micropolitan vs. Metropolitan',
    'area_type_2.0TRUE': 'Rural vs. Metropolitan',
    'ownership_2TRUE': 'Private vs. Government',
    'teaching_1.0TRUE': 'Non-teaching vs. Teaching',
    'size_2TRUE': 'Bedsize 100-300 vs. <100',
    'size_1TRUE': 'Bedsize >300 vs. <100',
    'income_household_median': 'Median household income',
    'disabled': 'Disabled, %',
    'education_less_highschool': 'Less than high school education, %',
    'hispanic': 'Hispanic, %',
    'race_black': 'Black, %',
    'female': 'Female, %',
    'age_over_65': 'Age ≥65, %'
}

variables_labels_updated = [label_mapping_updated.get(var, var) for var in variables]

# Create the forest plot
fig, ax = plt.subplots(figsize=(8, len(variables) * 0.5))

# Plot the estimates and confidence intervals
ax.errorbar(estimates, np.arange(len(estimates)), xerr=[estimates - ci_lower, ci_upper - estimates], fmt='o', color='black', ecolor='gray', capsize=3)
ax.axvline(x=1, linestyle='--', color='red')

# Set y-axis with variable names
ax.set_yticks(np.arange(len(variables_labels_updated)))
ax.set_yticklabels(variables_labels_updated)

# Set labels
ax.set_xlabel('Odds Ratios with 95% CI')

# Adjust x-axis limits and ticks
ax.set_xlim([0.1, 10])
ax.set_xscale('log')
ax.set_xticks([0.1, 1, 10])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Remove gridlines and y-axis title
ax.grid(False)
ax.set_ylabel('')  # Remove y-axis title

# Remove plot title
ax.set_title('')
fig2 = plt.gcf()
#%%
#%%
# Export
fig.savefig(path + 'forest_plot_new.png', format='png', bbox_inches='tight', dpi=300)
fig.savefig(path + 'forest_plot_new.svg', format='svg', bbox_inches='tight')

# %%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
# Load the CSV file
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/new_output_results/'
data = pd.read_csv(path + 'new_mlr_pdis.csv')

#%%
# Define the desired order of variables
# Define the desired order of variables based on the file
desired_order = [
    'us_regions_4TRUE', 'us_regions_3TRUE', 'us_regions_2TRUE', 'area_type_1.0TRUE', 'area_type_2.0TRUE',
    'ownership_2TRUE', 'teaching_1.0TRUE', 'size_2TRUE', 'size_1TRUE',
    'disabled', 'income_household_median', 'education_less_highschool',
    'hispanic', 'race_black', 'female', 'age_over_65'
]

# Reorder the dataframe based on the desired order
data = data.set_index('Unnamed: 0').loc[desired_order].reset_index()

# Prepare data for the forest plot
variables = data['Unnamed: 0']
estimates = data['OR']
ci_lower = data['OR_2.5_ci']
ci_upper = data['OR_97.5_ci']

# Create the forest plot
fig, ax = plt.subplots(figsize=(8, len(variables) * 0.5))

# Plot the estimates and confidence intervals
ax.errorbar(estimates, np.arange(len(estimates)), xerr=[estimates - ci_lower, ci_upper - estimates], fmt='o', color='black', ecolor='gray', capsize=3)
ax.axvline(x=0, linestyle='--', color='red')

# Set y-axis with variable names
ax.set_yticks(np.arange(len(variables)))
ax.set_yticklabels(variables)

# Set labels
ax.set_xlabel('Odds Ratio')


# Show plot
plt.tight_layout()
plt.show()

# %%
# Define the desired order of variables based on the file, excluding the intercept and inverted
desired_order = [
    'ownership_2TRUE', 'teaching_1.0TRUE', 'area_type_2.0TRUE', 'area_type_1.0TRUE', 'us_regions_2TRUE', 'us_regions_3TRUE', 'us_regions_4TRUE', 'size_1TRUE', 'size_2TRUE',
    'disabled', 'education_less_highschool', 'income_household_median',
    'hispanic', 'race_black', 'female', 'age_over_65'
]

# Reorder the dataframe based on the desired order
data = data.set_index('Unnamed: 0').loc[desired_order].reset_index()

# Prepare data for the forest plot
variables = data['Unnamed: 0']
estimates = data['OR']
ci_lower = data['OR_2.5_ci']
ci_upper = data['OR_97.5_ci']

# Map the new labels to the variables
label_mapping_updated = {
    'us_regions_4TRUE': 'Region, West vs. Northeast',
    'us_regions_3TRUE': 'Region, South vs. Northeast',
    'us_regions_2TRUE': 'Region, Midwest vs. Northeast',
    'area_type_1.0TRUE': 'Micropolitan vs. Metropolitan',
    'area_type_2.0TRUE': 'Rural vs. Metropolitan',
    'ownership_2TRUE': 'Private vs. Government',
    'teaching_1.0TRUE': 'Non-teaching vs. Teaching',
    'size_2TRUE': 'Bedsize 100-300 vs. <100',
    'size_1TRUE': 'Bedsize >300 vs. <100',
    'income_household_median': 'Median household income',
    'disabled': 'Disabled, %',
    'education_less_highschool': 'Less than high school education, %',
    'hispanic': 'Hispanic, %',
    'race_black': 'Black, %',
    'female': 'Female, %',
    'age_over_65': 'Age ≥65, %'
}

variables_labels_updated = [label_mapping_updated.get(var, var) for var in variables]

# Create the forest plot
fig, ax = plt.subplots(figsize=(8, len(variables) * 0.5))

# Plot the estimates and confidence intervals
ax.errorbar(estimates, np.arange(len(estimates)), xerr=[estimates - ci_lower, ci_upper - estimates], fmt='o', color='black', ecolor='gray', capsize=3)
ax.axvline(x=1, linestyle='--', color='red')

# Set y-axis with variable names
ax.set_yticks(np.arange(len(variables_labels_updated)))
ax.set_yticklabels(variables_labels_updated)

# Set labels
ax.set_xlabel('Odds Ratios with 95% CI')

# Adjust x-axis limits and ticks
ax.set_xlim([0.1, 10])
ax.set_xscale('log')
ax.set_xticks([0.1, 1, 10])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Remove gridlines and y-axis title
ax.grid(False)
ax.set_ylabel('')  # Remove y-axis title

# Remove plot title
ax.set_title('')
fig2 = plt.gcf()
#%%
#%%
# Export
fig.savefig(path + 'forest_plot_pdis.png', format='png', bbox_inches='tight', dpi=300)
fig.savefig(path + 'forest_plot_pdis.svg', format='svg', bbox_inches='tight')

# %%
#BURDEN HF AMI PATIENTS
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your dataset
df = pd.read_stata('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/merged_cms.dta')

#%%
# Define years and relevant columns
years = ['2018', '2019', '2020', '2021', '2022']
rpm_cols = [f"rpm_any{y[-2:]}" for y in years]
ami_cols = [f"discharges_ami_{y[-2:]}" for y in years]
hf_cols = [f"discharges_hf_{y[-2:]}" for y in years]

# Function to calculate discharge percentages
def get_discharge_percentage(discharge_cols, label):
    data = []
    for year, rpm_col, dis_col in zip(years, rpm_cols, discharge_cols):
        temp = df[[rpm_col, dis_col]].dropna()
        grouped = temp.groupby(temp[rpm_col]).agg(total_discharges=(dis_col, 'sum')).reset_index()
        total = grouped['total_discharges'].sum()
        grouped['percentage'] = grouped['total_discharges'] / total * 100
        grouped['RPM Status'] = grouped[rpm_col].map({0: 'No RPM', 1: 'RPM'})
        grouped['Year'] = year
        grouped['Condition'] = label
        data.append(grouped[['Year', 'Condition', 'RPM Status', 'percentage']])
    return pd.concat(data)

# Get data for AMI and HF
ami_pct = get_discharge_percentage(ami_cols, "Acute Myocardial Infarction")
hf_pct = get_discharge_percentage(hf_cols, "Heart Failure")
combined_pct = pd.concat([ami_pct, hf_pct])

# Plot
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Define desired legend and color order
rpm_order = ["RPM", "No RPM"]
colors = {"RPM": "lightblue", "No RPM": "gray"}

for i, condition in enumerate(["Heart Failure", "Acute Myocardial Infarction"]):
    subset = combined_pct[combined_pct['Condition'] == condition]
    pivot = subset.pivot(index='Year', columns='RPM Status', values='percentage')[rpm_order]
    
    # Stacked bar plot
    pivot.plot(kind='bar', stacked=True, ax=ax[i], color=[colors[k] for k in rpm_order], rot=0)
    ax[i].set_ylabel("Total Hospitalizations per Year, %")
    ax[i].set_title(f"Percentage of Hospitalizations for {condition}")
    ax[i].set_ylim(0, 100)
    ax[i].legend().remove()  # Remove subplot legend


    # Add percentage labels
    for p in ax[i].patches:
        height = p.get_height()
        if height > 3:
            ax[i].annotate(f'{height:.1f}%',
                           (p.get_x() + p.get_width() / 2, p.get_y() + height / 2),
                           ha='center', va='center', color='black', fontsize=11, fontweight='bold')

# Custom legend
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, title="RPM Availability", loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

# Final layout
plt.xlabel("Year")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

# %%
fig.savefig('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/rpm_discharge_plot.svg', format="svg", bbox_inches='tight')

#%%
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Create a long-format dataset for HF
hf_data = []

for year in range(2018, 2023):
    yr = str(year)[-2:]
    rpm_col = f"rpm_any{yr}"
    dis_col = f"discharges_ami_{yr}"

    temp = df[[rpm_col, dis_col]].dropna()
    for rpm_status in [0, 1]:
        total = temp[temp[rpm_col] == rpm_status][dis_col].sum()
        hf_data.append({'year': year, 'rpm': rpm_status, 'discharges': total})

hf_df = pd.DataFrame(hf_data)

# Create a binary indicator: 1 = discharges at RPM hospital, 0 = otherwise
hf_trend = hf_df.pivot(index='year', columns='rpm', values='discharges').fillna(0)
hf_trend = hf_trend.rename(columns={0: 'no_rpm', 1: 'with_rpm'})
hf_trend['total'] = hf_trend['no_rpm'] + hf_trend['with_rpm']
hf_trend = hf_trend.reset_index()

# Expand to individual rows for logistic regression
hf_expanded = []
for _, row in hf_trend.iterrows():
    hf_expanded.extend([{'year': row['year'], 'rpm': 1}] * int(row['with_rpm']))
    hf_expanded.extend([{'year': row['year'], 'rpm': 0}] * int(row['no_rpm']))

hf_expanded_df = pd.DataFrame(hf_expanded)

# Logistic regression to test for linear trend
model = smf.glm("rpm ~ year", data=hf_expanded_df, family=sm.families.Binomial()).fit()
print(model.summary())

# %%
import matplotlib.pyplot as plt
import os

#%%
# Load your dataset
df = pd.read_stata('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/merged_cms.dta')

#%%
# Define years and relevant columns
years = ['2018', '2019', '2020', '2021', '2022']
rpm_cols = [f"rpm_any{y[-2:]}" for y in years]
ami_cols = [f"discharges_ami_{y[-2:]}" for y in years]
hf_cols = [f"discharges_hf_{y[-2:]}" for y in years]

# Function to calculate percentage of discharges at RPM hospitals
def get_rpm_only_percentage(discharge_cols, label):
    data = []
    for year, rpm_col, dis_col in zip(years, rpm_cols, discharge_cols):
        temp = df[[rpm_col, dis_col]].dropna()
        total_rpm = temp[temp[rpm_col] == 1][dis_col].sum()
        total = temp[dis_col].sum()
        pct_rpm = (total_rpm / total) * 100 if total > 0 else 0
        data.append({'Year': int(year), 'Condition': label, 'Percent RPM': pct_rpm})
    return pd.DataFrame(data)

# Get data for AMI and HF
ami_pct = get_rpm_only_percentage(ami_cols, "AMI")
hf_pct = get_rpm_only_percentage(hf_cols, "HF")
combined_pct = pd.concat([ami_pct, hf_pct])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
for condition, group in combined_pct.groupby("Condition"):
    ax.plot(group['Year'], group['Percent RPM'], marker='o', label=condition)

# Formatting
ax.set_title("Percent of AMI and HF Discharges at RPM Hospitals (2018–2022)")
ax.set_xlabel("Year")
ax.set_ylabel("Discharges at RPM Hospitals (%)")
ax.set_xticks([int(y) for y in years])
ax.set_ylim(0, 100)
ax.legend(title="Condition")
ax.grid(False)

plt.tight_layout()
plt.show()

# Save figure
fig.savefig('/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/rpm_discharge_trend_only_rpm.svg', format="svg", bbox_inches='tight')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'merged_cms.dta')

# Filter data
hf_data = df[['rpm_any22', 'discharges_hf_22']].dropna()
ami_data = df[['rpm_any22', 'discharges_ami_22']].dropna()

# Plot setup
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

# Plot HF
sns.kdeplot(data=hf_data[hf_data['rpm_any22'] == 0], x='discharges_hf_22', ax=axes[0],
            linestyle='--', color='black', linewidth=2, label='No RPM')
sns.kdeplot(data=hf_data[hf_data['rpm_any22'] == 1], x='discharges_hf_22', ax=axes[0],
            linestyle='-', color='blue', linewidth=2, label='RPM')
axes[0].set_title('HF')
axes[0].set_ylabel('Density')
axes[0].legend()

# Plot AMI
sns.kdeplot(data=ami_data[ami_data['rpm_any22'] == 0], x='discharges_ami_22', ax=axes[1],
            linestyle='--', color='black', linewidth=2, label='No RPM')
sns.kdeplot(data=ami_data[ami_data['rpm_any22'] == 1], x='discharges_ami_22', ax=axes[1],
            linestyle='-', color='blue', linewidth=2, label='RPM')
axes[1].set_title('AMI')
axes[1].set_xlabel('Number of Discharges')
axes[1].set_ylabel('Density')
axes[1].legend()

# Final touches
for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig(path + 'rpm_discharge_distribution_kde_2022_fixed.svg', format='svg')
plt.show()

# %%
# Prepare datasets
hf_data = df[['rpm_any18', 'rpm_any22', 'discharges_hf_18', 'discharges_hf_22']].dropna()
ami_data = df[['rpm_any18', 'rpm_any22', 'discharges_ami_18', 'discharges_ami_22']].dropna()

# Plot setup
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

# Define fill and line styles
fill_colors = {0: 'lightgray', 1: 'lightblue'}
line_styles = {2018: '--', 2022: '-'}


# --- Heart Failure ---
for year in [2018, 2022]:
    rpm_col = f'rpm_any{str(year)[-2:]}'
    dis_col = f'discharges_hf_{str(year)[-2:]}'
    for rpm_status in [0, 1]:
        subset = hf_data[hf_data[rpm_col] == rpm_status][dis_col]
        label = f"{year} - {'RPM' if rpm_status == 1 else 'No RPM'}"
        sns.kdeplot(
            data=subset,
            ax=axes[0],
            fill=True,
            common_norm=False,
            alpha=0.6,
            color=fill_colors[rpm_status],
            linestyle=line_styles[year],
            linewidth=2,
            label=label
        )

axes[0].set_title('Distribution of Heart Failure Hospitalizations')
axes[0].set_xlabel('Number of Hospitalizations')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(False)

# --- AMI ---
for year in [2018, 2022]:
    rpm_col = f'rpm_any{str(year)[-2:]}'
    dis_col = f'discharges_ami_{str(year)[-2:]}'
    for rpm_status in [0, 1]:
        subset = ami_data[ami_data[rpm_col] == rpm_status][dis_col]
        label = f"{year} - {'RPM' if rpm_status == 1 else 'No RPM'}"
        sns.kdeplot(
            data=subset,
            ax=axes[1],
            fill=True,
            common_norm=False,
            alpha=0.6,
            color=fill_colors[rpm_status],
            linestyle=line_styles[year],
            linewidth=2,
            label=label
        )

axes[1].set_title('Distribution of Acute Myocardial Infarction Hospitalizations')
axes[1].set_xlabel('Number of Hospitalizations')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(False)

# Set common x-axis range for both plots: 0 to 2500 discharges
for ax in axes:
    ax.set_xlim(-250, 2500)
    ax.set_xticks(range(-250, 2501, 500))  # ticks every 500

# Save and show
plt.tight_layout()
plt.savefig(path + 'rpm_discharge_distribution_kde_2018_2022_filled.svg', format='svg')
plt.show()
# %%
# MAP DISTRIBUTION
#%%
import geopandas as gpd
import pandas as pd

#%%
# Define file paths
zip_shapefile = "/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/AHA_RPM_map/tl_2022_us_zcta520/tl_2022_us_zcta520.shp"
state_shapefile = "/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/AHA_RPM_map/tl_2022_us_state/tl_2022_us_state.shp"
county_shapefile = "/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/AHA_RPM_map/tl_2022_us_county/tl_2022_us_county.shp"
data_path = "/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/aha_18-22_final_dataset.dta"

#%%
# Load shapefiles
zip_gdf = gpd.read_file(zip_shapefile)
state_gdf = gpd.read_file(state_shapefile)
county_gdf = gpd.read_file(county_shapefile)
aha_data = pd.read_stata(data_path)

#%%
# Define FIPS codes for non-continental areas
non_continental_fips = {"02", "15", "72", "66", "60", "69", "78"}  # Alaska, Hawaii, PR, territories

#%%
# Remove non-continental areas from all datasets
state_gdf = state_gdf[~state_gdf["STATEFP"].isin(non_continental_fips)].copy()
county_gdf = county_gdf[~county_gdf["STATEFP"].isin(non_continental_fips)].copy()
zip_gdf = zip_gdf[zip_gdf["ZCTA5CE20"].astype(str).str[:2].isin(set(state_gdf["STATEFP"].unique()))].copy()

#%%
# Ensure geometry column is set correctly
state_gdf = state_gdf.set_geometry("geometry")
county_gdf = county_gdf.set_geometry("geometry")
zip_gdf = zip_gdf.set_geometry("geometry")

#%%
# Prepare ZIP data
aha_data["zipcode"] = aha_data["mloczip"].str.extract(r'(\d+)').astype(float)
zip_gdf["zipcode"] = zip_gdf["ZCTA5CE20"].astype(float)

#%%
# Add COUNTYFP information to zip_gdf
zip_gdf = gpd.sjoin(zip_gdf, county_gdf[["COUNTYFP", "geometry"]], how="left", predicate="intersects")

#%%
# Merge ZIP code shapefile with RPM data
merged_gdf = zip_gdf.merge(aha_data, on="zipcode", how="left")

#%%
# Aggregate RPM data by county
rpm_by_county = merged_gdf.groupby("COUNTYFP").agg(
    rpm_count=("rpm_any22", "sum"),  # Sum of hospitals with RPM=yes
    total_count=("rpm_any22", "count")  # Total ZIP codes in the county
).reset_index()

#%%
# Merge RPM data back with county shapefile
county_gdf = county_gdf.merge(rpm_by_county, on="COUNTYFP", how="left")

#%%
# Normalize RPM count for color intensity
county_gdf["intensity"] = county_gdf["rpm_count"].fillna(0) / county_gdf["total_count"].fillna(1)

#%%
# Plot the map
fig, ax = plt.subplots(figsize=(12, 8))
county_gdf.plot(ax=ax, column="intensity", cmap="Greens", edgecolor="gray", linewidth=0.3, legend=True)  # Counties colored by RPM intensity
state_gdf.plot(ax=ax, color="none", edgecolor="gray", linewidth=0.5)  # State boundaries

# Remove axes and set title
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)
ax.set_title("Map of RPM Intensity by County in the Continental US", fontsize=14)

# Save the map
plt.savefig("map_rpm_intensity.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
#SENSITIVITY RUCA CODES
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_merged_ruca.dta')

# Define RUCA mapping
ruca_labels = {1: 'Metropolitan', 2: 'Micropolitan', 3: 'Small Town', 4: 'Rural'}

# Define years and relevant columns
years = [2018, 2019, 2020, 2021, 2022]
rpm_cols = [f"rpm_any{str(y)[-2:]}" for y in years]

# Initialize storage for results
plot_data = []

# Loop through years and RUCA categories
for year, rpm_col in zip(years, rpm_cols):
    temp = df[[rpm_col, 'ruca']].dropna()
    for ruca_code in [1, 2, 3, 4]:
        subset = temp[temp['ruca'] == ruca_code]
        if len(subset) == 0:
            pct_rpm = 0
        else:
            pct_rpm = subset[rpm_col].mean() * 100
        plot_data.append({'Year': year, 'RUCA': ruca_labels[ruca_code], 'RPM %': pct_rpm})

# Create DataFrame
plot_df = pd.DataFrame(plot_data)

# Set the desired column order
ruca_order = ['Metropolitan', 'Micropolitan', 'Small Town', 'Rural']

# Pivot and reorder columns
pivot_df = plot_df.pivot(index='Year', columns='RUCA', values='RPM %')[ruca_order]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
pivot_df.plot(kind='bar', ax=ax)

# Formatting
ax.set_title('Percentage of Hospitals with RPM by RUCA Code (2018–2022)')
ax.set_ylabel('Hospitals with RPM (%)')
ax.set_xlabel('Year')
ax.set_ylim(0, 100)
ax.legend(title='RUCA Category', loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

# Save figure
plt.savefig(path + 'rpm_by_ruca_barplot.svg', format='svg')
plt.show()
# %%



#%%
#Line plot
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_merged_ruca.dta')

# Define RUCA mapping and order
ruca_labels = {1: 'Metropolitan', 2: 'Micropolitan', 3: 'Small Town', 4: 'Rural'}
ruca_order = ['Metropolitan', 'Micropolitan', 'Small Town', 'Rural']

# Define years and RPM columns
years = [2018, 2019, 2020, 2021, 2022]
rpm_cols = [f"rpm_any{str(y)[-2:]}" for y in years]

# Collect percentage data
plot_data = []

for year, rpm_col in zip(years, rpm_cols):
    temp = df[[rpm_col, 'ruca']].dropna()
    for ruca_code in [1, 2, 3, 4]:
        subset = temp[temp['ruca'] == ruca_code]
        pct_rpm = subset[rpm_col].mean() * 100 if len(subset) > 0 else 0
        plot_data.append({'Year': year, 'RUCA': ruca_labels[ruca_code], 'RPM %': pct_rpm})

# Create DataFrame and pivot
plot_df = pd.DataFrame(plot_data)
pivot_df = plot_df.pivot(index='Year', columns='RUCA', values='RPM %')[ruca_order]

# Plot line chart
fig, ax = plt.subplots(figsize=(10, 6))
pivot_df.plot(kind='line', marker='o', ax=ax)

# Format
ax.set_title('Percentage of Hospitals with RPM by RUCA Code (2018–2022)')
ax.set_ylabel('Hospitals with RPM (%)')
ax.set_xlabel('Year')
ax.set_ylim(0, 100)
ax.legend(title='RUCA Category', loc='upper left')
ax.grid(False)
plt.xticks(years)  # Ensure proper x-tick labels
plt.tight_layout()

# Save figure
plt.savefig(path + 'rpm_by_ruca_lineplot.svg', format='svg')
plt.show()

# %%
import statsmodels.formula.api as smf
import pandas as pd

# Ensure categorical encoding
df['ruca'] = pd.Categorical(df['ruca'], categories=['Metropolitan', 'Micropolitan', 'Small Town', 'Rural'], ordered=False)
df['size'] = df['size'].astype('category')
df['us_region'] = df['us_regions'].astype('category')  # or 'region' if that's the correct column
df['teaching'] = df['teaching'].astype('category')
df['ownership'] = df['ownership'].astype('category')

# Fit adjusted ANCOVA-like model
model = smf.ols(
    'Q("RPM %") ~ Year * ruca + size + us_regions + teaching + ownership',
    data=df
).fit()

# Print results
print(model.summary())

# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Assuming you already have this from earlier
# plot_df has columns: 'Year', 'RUCA', 'RPM %'

# Ensure RUCA is categorical in correct order
plot_df['RUCA'] = pd.Categorical(plot_df['RUCA'], categories=['Metropolitan', 'Micropolitan', 'Small Town', 'Rural'])

# Fit ANCOVA-style model
model = smf.ols('Q("RPM %") ~ Year * RUCA', data=plot_df).fit()

# Create a new DataFrame with all Year–RUCA combinations for prediction
prediction_df = pd.DataFrame([(year, ruca) for year in range(2018, 2023) for ruca in plot_df['RUCA'].cat.categories],
                             columns=['Year', 'RUCA'])

# Predict fitted values
prediction_df['Predicted RPM %'] = model.predict(prediction_df)

# Plot original data
fig, ax = plt.subplots(figsize=(10, 6))
for ruca in plot_df['RUCA'].cat.categories:
    subset = plot_df[plot_df['RUCA'] == ruca]
    ax.plot(subset['Year'], subset['Q("RPM %")' if "Q(" in subset.columns[2] else "RPM %"], marker='o', linestyle='-', label=f"{ruca} (Observed)")

# Plot fitted lines
for ruca in plot_df['RUCA'].cat.categories:
    pred = prediction_df[prediction_df['RUCA'] == ruca]
    ax.plot(pred['Year'], pred['Predicted RPM %'], linestyle='--', linewidth=2, label=f"{ruca} (Fitted)")

# Format plot
ax.set_title('Observed vs Fitted RPM % Trends by RUCA (2018–2022)')
ax.set_xlabel('Year')
ax.set_ylabel('Hospitals with RPM (%)')
ax.set_ylim(0, 100)
ax.grid(True)
ax.legend(loc='upper left', title='RUCA Category')

plt.tight_layout()
plt.savefig(path + 'rpm_ruca_trend_with_fitted.svg', format='svg')
plt.show()


# %%
#ADJUSTED ANCOVA ANALYSIS
import pandas as pd
import statsmodels.formula.api as smf

# Load the dataset
path = '/Users/af955/Library/CloudStorage/Box-Box/Aline_Rohan/AHA_data/new_analysis_05.2025/'
df = pd.read_stata(path + 'aha_merged_ruca.dta')  # <- update filename if needed

# Reshape to long format
rpm_cols = ['rpm_any18', 'rpm_any19', 'rpm_any20', 'rpm_any21', 'rpm_any22']
df_long = df.melt(
    id_vars=['ruca', 'size', 'us_regions', 'teaching', 'ownership'],
    value_vars=rpm_cols,
    var_name='year',
    value_name='rpm_available'
)

# Extract numeric year
df_long['year'] = df_long['year'].str.extract(r'(\d+)').astype(int)

# Convert to categorical
df_long['ruca'] = pd.Categorical(df_long['ruca'], categories=[1, 2, 3, 4], ordered=False)
df_long['size'] = df_long['size'].astype('category')
df_long['us_regions'] = df_long['us_regions'].astype('category')
df_long['teaching'] = df_long['teaching'].astype('category')
df_long['ownership'] = df_long['ownership'].astype('category')

# Fit ANCOVA-style model
model = smf.ols(
    'rpm_available ~ year * ruca + size + us_regions + teaching + ownership',
    data=df_long
).fit()

# Print summary
print(model.summary())

# %%
