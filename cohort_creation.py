import pandas as pd
import numpy as np

# Load your dataset
path = "/mnt/data/your_dataset.dta"  # update this path as needed
df = pd.read_stata(path)

# --- US Regions ---
df['us_regions'] = pd.NA
region_map = {
    1: ["CT", "ME", "MA", "NJ", "NY", "PA", "RI", "VT", "NH"],
    2: ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    3: ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV", "DC"],
    4: ["AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"]
}
for k, states in region_map.items():
    df.loc[df['mstate'].isin(states), 'us_regions'] = k

# --- Composite Variables ---
df['rpm_hosp'] = df['pdishos']
df['rpm_sys'] = df['pdissys']
df['pport_access'] = df['pefviis'] + df['pefvios']
df['down_mr'] = df['pefdiis'] + df['pefdios']
df['imp_ext_mr'] = df['pefiris'] + df['pefiros']
df['exp_mr'] = df['pefsfis'] + df['pefsfos']
df['upd_amend_mr'] = df['pefrais'] + df['pefraos']
df['view_clnotes'] = df['pefvcnis'] + df['pefvcnos']
df['access_ehr_app'] = df['pefapiis'] + df['pefapios']
df['app_fhir'] = df['peffhiis'] + df['peffhios']
df['share_data'] = df['pefsdis'] + df['pefsdos'] + df['pefapis'] + df['pefapos']
df['msg_prov'] = df['persmis'] + df['persmos']

# --- Ownership ---
df['cont_owen'] = df['mcntrl']
df.loc[df['cont_owen'].isin([12,13,14,15,16]), 'cont_owen'] = 1
df.loc[df['cont_owen'].isin([21,23]), 'cont_owen'] = 2
df.loc[df['cont_owen'].isin([31,32,33]), 'cont_owen'] = 3
df.loc[df['cont_owen'].isin([40,44,45,46,47,48]), 'cont_owen'] = 4

df['ownership'] = pd.NA
df.loc[df['cntrl'].isin([12,13,14,15,16,40,44,45,46,47,48]), 'ownership'] = 1
df.loc[df['cntrl'].isin([31,32,33,21,23]), 'ownership'] = 2

# --- Size ---
df['size'] = pd.NA
df.loc[df['bdtot'] < 100, 'size'] = 1
df.loc[(df['bdtot'] >= 100) & (df['bdtot'] <= 300), 'size'] = 2
df.loc[df['bdtot'] > 300, 'size'] = 3

# --- Composite variable ---
df['composite'] = df['imp_ext_mr'] + df['exp_mr'] + df['upd_amend_mr'] + df['access_ehr_app'] + df['share_ipop']

# --- Area Type ---
df['area_type'] = pd.NA
df.loc[df['cbsatype'] == 'Metro', 'area_type'] = 0
df.loc[df['cbsatype'] == 'Micro', 'area_type'] = 1
df.loc[df['cbsatype'] == 'Rural', 'area_type'] = 2

# --- Teaching ---
df['teaching'] = df['mapp8']

# --- ERR Variables ---
for window in ['19_22', '18_21', '17_20', '16_19', '15_18']:
    for cond in ['hf', 'ami']:
        var = f'err{cond}_{window}'
        df[f'{var}_d'] = np.where(df[var] <= 1, 1, np.where(df[var] > 1, 0, np.nan))

# --- RPM Function ---
def rpm_logic(df, prefix):
    cond_1 = (df[f'{prefix}hos'] == 1) | (df[f'{prefix}sys'] == 1) | (df[f'{prefix}ven'] == 1)
    cond_0 = (df[f'{prefix}hos'] == 0) | (df[f'{prefix}sys'] == 0) | (df[f'{prefix}ven'] == 0)
    return np.where(cond_1, 1, np.where(cond_0 & ~cond_1, 0, np.nan))

# Apply to 2018
df['rpm_pdis18'] = rpm_logic(df, 'pdis')
df['rpm_chro18'] = rpm_logic(df, 'chcar')
df['rpm_other18'] = rpm_logic(df, 'orpm')
df['rpm_any18'] = np.where(
    df[['rpm_pdis18', 'rpm_chro18', 'rpm_other18']].eq(1).any(axis=1), 1,
    np.where(df[['rpm_pdis18', 'rpm_chro18', 'rpm_other18']].eq(0).all(axis=1), 0, np.nan)
)

# Ever RPM
df['rpm_ever'] = 0
df.loc[df[['rpm_any18', 'rpm_any19', 'rpm_any20', 'rpm_any21', 'rpm_any22']].eq(1).any(axis=1), 'rpm_ever'] = 1

# Apply to 2022
df['rpm_pdis22'] = rpm_logic(df, 'pdis')
df['rpm_chro22'] = rpm_logic(df, 'chcar')
df['rpm_other22'] = rpm_logic(df, 'orpm')
df['rpm_any22'] = np.where(
    df[['rpm_pdis22', 'rpm_chro22', 'rpm_other22']].eq(1).any(axis=1), 1,
    np.where(df[['rpm_pdis22', 'rpm_chro22', 'rpm_other22']].eq(0).all(axis=1), 0, np.nan)
)

# --- Critical Access ---
df['is_critical_access'] = (df['SiteSubcategory'] == 'Critical Access Hospitals').astype(int)

# --- Merge key ---
df['city_state'] = df['city_ascii'] + ', ' + df['state_id']
