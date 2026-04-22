import pandas as pd
import numpy as np
import joblib
import gc
import os
import io
import math
import re
from sklearn.preprocessing import StandardScaler
import shap
pd.set_option('future.no_silent_downcasting', True)

# Flattening the data

#flatenning json
def CPTransformer1(data):

    def flatten_json(Y):
        out = {}

        def flatten(X, name=''):
            if isinstance(X, dict):
                for a in X:
                    flatten(X[a], name + a + '_')
            elif isinstance(X, list):
                i = 0
                for a in X:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = X

        flatten(Y)
        return out

    flat = flatten_json(data)
    my_list = pd.json_normalize(flat)

    my_list['leadDetail_Created_Date']=pd.to_datetime(my_list['leadDetail_Created_Date'], format="%Y-%m-%d")
    #my_list['leadDetail_Re-engagement_count']= my_list.filter(regex=("reEnquiries_(.*)_Id")).shape[1]

    leadid = my_list['leadDetail_Enquiry_Ref_No_']
    Enquiryrefno = my_list['leadDetail_Lead_ID']
    my_list['leadDetail_Project_Interested'] =my_list['leadDetail_Project_Interested'].str.replace('Â', '', regex=False).str.replace('\xa0', ' ', regex=False)
    return my_list

def clean_text(text):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return text
    return re.sub(r'[^\x00-\x7F]+', '', str(text)).strip().lower()    
    
def cp_pipeline_withoutQ(df, df_cyclic, df_inv):
        #Rename columns to standardized names
    CP_COLUMN_MAPPING = {
     'leadDetail_Re-engagement_count': 'Re-engagement count',
     'quotation_Quotation_count':'Quotation',
     'AIMLObject_Total_Incoming_Answered_Calls__c':'Total Incoming Answered Calls',
     'AIMLObject_Total_Outgoing_Answered_Calls__c':'Total Outgoing Answered Calls',
     'Accounts_Account Name':'CP_Name',
     'Accounts_TEAM SIZE_STATUS':'TEAM SIZE_STATUS',
     'Accounts_CP Type':'CP Type',
     'Accounts_TARGET SALE FOR THIS FINANCIAL YEAR_STATUS':'TARGET SALE FOR THIS FINANCIAL YEAR_STATUS',
     'Accounts_INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS':'INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS',
     'Accounts_WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1':'WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1',
     'Accounts_WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1':'WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1',
     'Accounts_Club Type_STATUS':'Club Type_STATUS',
     'Accounts_DEVELOPERS YOU WORK WITH_STATUS_1':'DEVELOPERS YOU WORK WITH_STATUS_1',
     'leadDetail_Stage':'Stage',
     'leadDetail_Site_Visit_Form':'Site Visit Form',
     'leadDetail_Opportunity_Name':'Opportunity Name',
     'leadDetail_Opportunity_Owner':'Opportunity Owner',
     'leadDetail_Lead_Owner':'Lead Owner',
     'leadDetail_Lead_Status':'Lead Status',
     'leadDetail_Converted_Date':'Converted Date',
     'leadDetail_Company_/_Account':'Company / Account',
     'leadDetail_CP_Agent':'CP Agent',
     'leadDetail_CP_Name':'CP Name_lead',
     'leadDetail_AI/ML_Object':'AI/ML Object',
     'leadDetail_Category':'Category',
     'leadDetail_SFT_Range':'SFT Range',
     'leadDetail_Interested_in':'Interested in',
     'leadDetail_Applicant_-_Nature_of_Occupation':'Applicant - Nature of Occupation',
     'leadDetail_Preferred_Property_Type':'Preferred Property Type',
     'leadDetail_Medium':'Medium',
     'leadDetail_Market_/_Channel':'MarketChannel',
     'leadDetail_Project_Interested':'Project_Interested',
     'leadDetail_Appointment_Initiated_Date':'Appointment_Initiated_Date',
     'leadDetail_Refer_a_family_or_friend':'Refer a family or friend',
     'leadDetail_Budget':'Budget',
     'leadDetail_Reason_for_Purchase':'Reason for Purchase',
     'leadDetail_Is_this_your_First_Home_Purchase?':'Is this your First Home Purchase?',
     'leadDetail_Household_Monthly_Income_(Rs)':'Household Monthly Income (Rs)',
     'leadDetail_Current_Accomodation_Type':'Current Accomodation Type',
     'leadDetail_Applicant_-_Work_Location':'Applicant - Work Location',
     'leadDetail_Applicant_-_Industry':'Applicant - Industry',
     'leadDetail_How_Many_Kids?':'How Many Kids?',
     'leadDetail_Marital_Status':'Marital Status',
     'leadDetail_Address_(Website)':'Address (Website)',
     'leadDetail_Age':'Age',
     'leadDetail_Gender':'Gender',
     'leadDetail_Enquiry_Ref_No_':'Enquiry Ref No.',
     'leadDetail_Created_Date':'Created Date',
     'leadDetail_Lead_ID':'Lead ID',
     'leadDetail_Appointment Date':'Appointment Date',
     'leadDetail_Call_remarks':'Call_remarks',
     }
    # Step 1: Rename columns immediately after input
    df.rename(columns=CP_COLUMN_MAPPING, inplace=True)
    df_cyclic.rename(columns={'CP Name': 'CP_Name'}, inplace=True)
    #df['CP Name']=df['CP Name'].str.lower()
    # --- Drop Unnecessary Columns ---
    df.drop(
        columns=[
            'CP Agent', 'Lead Status', 'Address (Website)', 'Sub Source',
            'Opportunity Name', 'MarketChannel', 'Company / Account'
        ],
        errors='ignore',
        inplace=True
    )

    #---------- INVENTORY FEATURE ENGINEERING STARTS HERE ----------

    # Clean project columns (vectorized)
    df['project_clean'] = df['Project_Interested'].apply(clean_text)
    df_inv['project_clean'] = df_inv['Project Name'].apply(clean_text)

    # Consolidated datetime parsing once
    df['Appointment_Initiated_Date'] = pd.to_datetime(df['Appointment_Initiated_Date'], format="%Y-%m-%d", errors='coerce')
    df['Appointment Date'] = pd.to_datetime(df.get('Appointment Date'), format='%Y-%m-%d', errors='coerce')

    df_inv['Opening month'] = pd.to_datetime(df_inv['Opening month'], dayfirst=True, errors='coerce')
    df_inv['Project Launch Date'] = pd.to_datetime(df_inv['Project Launch Date'], dayfirst=True, errors='coerce')
    
    # Normalize to month start (typed keys, no string concat)
    df['month_start'] = df['Appointment_Initiated_Date'].dt.to_period('M').dt.to_timestamp()
    df_inv['opening_month_start'] = df_inv['Opening month'].dt.to_period('M').dt.to_timestamp()

    #print("before merging with inventory:", df.shape)
    # Direct multi-key merge instead of concatenated string key
    # df = df.merge(
    #     df_inv[['project_clean', 'opening_month_start', 'Sum of Unsold inventory']],
    #     left_on=['project_clean', 'month_start'],
    #     right_on=['project_clean', 'opening_month_start'],
    #     how='left',
    #     sort=False
    # ).drop(columns=['opening_month_start'])
    #print("after dfmerge:",df.shape)
    
    df['month_start'] = pd.to_datetime(df['month_start'])
    df_inv['opening_month_start'] = pd.to_datetime(df_inv['opening_month_start'])

    # Sort EXACTLY like this
    df = df.sort_values('month_start').reset_index(drop=True)
    df_inv = (
        df_inv
        .sort_values('opening_month_start')
        .drop_duplicates(['project_clean', 'opening_month_start'], keep='last')
        .sort_values('opening_month_start')
        .reset_index(drop=True)
    )

    # Perform asof merge (backward = last available previous value)
    df = pd.merge_asof(
        df,
        df_inv[['project_clean', 'opening_month_start', 'Sum of Unsold inventory']],
        left_on='month_start',
        right_on='opening_month_start',
        by='project_clean',
        direction='backward'   # 🔑 this is the key
    )

    # Optional: drop column if not needed
    df = df.drop(columns=['opening_month_start'])

    # Merge launch metadata once (per project)
    launch_meta = (
        df_inv[['project_clean', 'Project Launch Date', 'Project Launch inventory']]
        .drop_duplicates('project_clean')
    )
    df = df.merge(launch_meta, on='project_clean', how='left', suffixes=('', '_launchmeta'))

    # Vectorized fill for missing inventory using launch inventory when date condition holds
    need_fill = df['Sum of Unsold inventory'].isna()
    cond_fill = need_fill & (df['month_start'] <= df['Project Launch Date'])
    df.loc[cond_fill, 'Sum of Unsold inventory'] = df.loc[cond_fill, 'Project Launch inventory']

    #print("before dropping NA unsold inventory:",df.shape)
    
    # Drop rows still missing inventory (as before)
    df = df.dropna(subset=['Sum of Unsold inventory'])
    #print("after dropping NA unsold inventory:",df.shape)
    
    # Create Unsold Inventory % and replace original columns
    # Note: if Project Launch inventory can be zero, protect division
    denom = df['Project Launch inventory'].replace({0: np.nan})
    df['Unsold Inv perc'] = (df['Sum of Unsold inventory'] / denom) * 100
    df['Unsold Inv perc'] = df['Unsold Inv perc'].round(2)
    df.drop(columns=['Sum of Unsold inventory', 'Project Launch inventory'], inplace=True)
    df.rename(columns={'Unsold Inv perc': 'Sum of Unsold inventory'}, inplace=True)

    # Launch/Sustenance Feature (vectorized)
    has_launch = df['Project Launch Date'].notna()
    df['Launch_Sustenance'] = np.where(
        has_launch & (df['Appointment_Initiated_Date'] <= df['Project Launch Date']),
        'launch',
        'sustenance'
    )

    # Difference in days (vectorized)
    df['Launch_to_Appointment_Days'] = (df['Project Launch Date'] - df['Appointment_Initiated_Date']).dt.days

    # Binning Launch_to_Appointment (same as original)
    bins = [-np.inf, -180, -91, -61, -31, 0, 31, 61, 91, 180, np.inf]
    labels = [
        "-180+",
        "-180 to -91",
        "-91 to -61",
        "-61 to -31",
        "-31 to -1",
        "0-30",
        "31-60",
        "61-90",
        "91-180",
        "180+"
    ]
    df["Binned_Launch_to_Appointment"] = pd.cut(
        df["Launch_to_Appointment_Days"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    # Appointment initiated day/hour
    df['AI_day'] = df['Appointment_Initiated_Date'].dt.day_name()
    df['AI_hour'] = df['Appointment_Initiated_Date'].dt.hour

    # SV duration (vectorized)
    df['Appointment Date'] = (pd.to_datetime(df['Appointment Date'], errors='coerce').fillna(pd.to_datetime(df['Appointment_Initiated_Date'], errors='coerce')))
    df['SV_Duration'] = (df['Appointment_Initiated_Date'] - df['Appointment Date']).dt.days

    # Dropping unrequired columns
    columns_to_drop = ['Converted Date', 'Stage', 'Applicant - Work Location', 'Appointment Date']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # # Merge df and df_cp_info - wont be required in production as its in JSON
    # df['CP_Name_clean'] = df['CP_Name'].apply(clean_text)
    # df_cp_info['CP_Name_clean'] = df_cp_info['Account Name'].apply(clean_text)
    # df['CP_Name_clean'] = df['CP_Name_clean'].fillna('').astype(str)
    # df_cp_info['CP_Name_clean'] = df_cp_info['CP_Name_clean'].fillna('').astype(str)

    # df = df.merge(df_cp_info, on='CP_Name_clean', how='left', suffixes=('', '_cpinfo'))

    # print("After merge with df_cp_info:", df.shape)
    # # if 'Applicant - Nature of Occupation' in df.columns:
    # #     print("Nulls in a known CP column:", df['Applicant - Nature of Occupation'].isnull().sum())

    df_merged = df.copy()
    
    # Remove specified columns from df_merged
    df_merged.drop(columns=['RERA NUMBER_STATUS', 'CST NUMBER_STATUS'], errors='ignore', inplace=True)

    # Missing values imputation for CP info data
    impute_values_cp = {
        'CP Type': 'Domestic-CP',
        'TARGET SALE FOR THIS FINANCIAL YEAR_STATUS': 'above10-blw40',
        'TEAM SIZE_STATUS': 'less_than_eqlto_10',
        'INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS': 'No',
        'WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1': '0-2',
        'WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1': '0to3',
        'Club Type_STATUS': 'Club_Type_1',
        'ACCOUNT TYPE_STATUS': 'Account_Type_1',
        'SV_Duration': 30.5,
        'DEVELOPERS YOU WORK WITH_STATUS_1': '0-9'
    }
    for column, value in impute_values_cp.items():
        if column in df_merged.columns:
            df_merged[column] = df_merged[column].fillna(value)
    
    
    #Remove rows where 'CP Name' is missing
    if 'CP_Name' in df_merged.columns:
        df_merged = df_merged[df_merged['CP_Name'].notna()]
    
    df_merged = df_merged[df_merged['CP_Name'].notna()]    
    #print("Second check",df_merged.shape)
    
    # # Safely compute mode/median and fill missing values
    # if 'AI_day' in df_merged.columns and not df_merged['AI_day'].dropna().empty:
    #     ai_day_mode = df_merged['AI_day'].mode()[0]
    #     df_merged['AI_day'] = df_merged['AI_day'].fillna(ai_day_mode)

    # if 'AI_hour' in df_merged.columns and not df_merged['AI_hour'].dropna().empty:
    #     ai_hour_median = df_merged['AI_hour'].median()
    #     df_merged['AI_hour'] = df_merged['AI_hour'].fillna(ai_hour_median)

    # remove Created Date
    df_merged.drop(['Created_Date'], axis=1, errors='ignore', inplace=True)
    
    # Treating the missing values for Site Visit form data variables
    impute_values_form = {
        'Gender': 'Male',
        'Age': 42.0,
        'Marital Status': 'Married',
        'How Many Kids?': 0,
        'Applicant - Industry': 'IT/ Software',
        'Current Accomodation Type': 'self-own-house',
        'Household Monthly Income (Rs)': 0,
        'Is this your First Home Purchase?': 'no',
        'Reason for Purchase': 'self-use',
        'Budget': '70 - 80 Lakh',
        'Refer a family or friend': 'No',
        #'Re-engagement count': 0.0,
        'Total Incoming Answered Calls': 0.0,
        'Total Outgoing Answered Calls': 2.0,
        'Applicant - Nature of Occupation': 'Salaried Employee',
        'Interested in': 'Apartments',
        'SFT Range': '1001 - 1250'
    }
    for column, value in impute_values_form.items():
        if column in df_merged.columns:
            df_merged[column] = (df_merged[column].fillna(value).infer_objects(copy=False))

    # Remove the specified columns
    df_merged.drop(columns=['Medium', 'Source', 'Third Outgoing call Duration'], errors='ignore', inplace=True)
    
    # ---------------- Vectorized data binning & cleaning ----------------

    # Budget normalization via dictionary map (vectorized)
    budget_value_to_bucket = {
        '<30 Lakh': 'Less than 50 Lakhs',
        '<50 Lakh': 'Less than 50 Lakhs',
        '30 - 40 Lakh': 'Less than 50 Lakhs',
        '40 - 50 Lakh': 'Less than 50 Lakhs',
        '50 - 60 Lakh': '50 Lakhs - 75 Lakhs',
        '60 - 70 Lakh': '50 Lakhs - 75 Lakhs',
        '70 - 80 Lakh': '50 Lakhs - 75 Lakhs',
        '50 Lakhs - 75 Lakhs': '50 Lakhs - 75 Lakhs',
        '75 Lakhs - 1 Crore': '75 Lakhs - 1 Crore',
        '80 - 90 Lakh': '75 Lakhs - 1 Crore',
        '90 Lakh - 1 Crore': '75 Lakhs - 1 Crore',
        '1 - 1.2 Crore': '1 Crore - 1.5 Crores',
        '1.2 - 1.4 Crore': '1 Crore - 1.5 Crores',
        '1 Crore - 1.5 Crores': '1 Crore - 1.5 Crores',
        '1.4 - 1.6 Crore': '1.5 Crores - 2 Crores',
        '1.5 Crores - 2 Crores': '1.5 Crores - 2 Crores',
        '1.6 - 1.8 Crore': '1.5 Crores - 2 Crores',
        '1.8 - 2 Crore': '1.5 Crores - 2 Crores',
        '1.50 cr - 2 cr': '1.5 Crores - 2 Crores',
        '2 - 2.2 Crore': '2 Crores - 3 Crores',
        '2.2 - 2.4 Crore': '2 Crores - 3 Crores',
        '2.4 - 2.6 Crore': '2 Crores - 3 Crores',
        '2.6 - 2.8 Crore': '2 Crores - 3 Crores',
        '2.8 - 3 Crore': '2 Crores - 3 Crores',
        '2 Crores - 3 Crores': '2 Crores - 3 Crores',
        '3 - 3.5 Crore': '3 Crores - 4 Crores',
        '3.5 - 4 Crore': '3 Crores - 4 Crores',
        '3 Crores - 4 Crores': '3 Crores - 4 Crores',
        '4 - 4.5 Crore': '4 Crores - 5 Crores',
        '4.5 - 5 Crore': '4 Crores - 5 Crores',
        '4 Crores - 5 Crores': '4 Crores - 5 Crores',
        '5 Crores - 7.5 Crores': '5 Crores - 7.5 Crores',
        '10 Crores - 15 Crores': '10 Crores - 15 Crores',
        'Not Revealed': 'Unknown'
    }
    df_merged['Budget'] = df_merged['Budget'].map(budget_value_to_bucket).fillna('Unknown')

    # Replace 'Unknown' with mode if available
    filtered_budget = df_merged.loc[df_merged['Budget'] != 'Unknown', 'Budget']

    if not filtered_budget.empty:
        mode_value = filtered_budget.mode().iloc[0]
        df_merged['Budget'] = df_merged['Budget'].mask(
            df_merged['Budget'] == 'Unknown',
            mode_value
        )

    # Preferred Property Type mapping (vectorized)
    map_pt = {
        'plots': 'Plots', '2bhk': '2BHK', '3 bhk 3t': '3BHK', '3 bhk 2t': '3BHK',
        '1bhk': '1BHK', '3 bhk': '3BHK', '2.5 bhk': '2BHK', '4 bhk': '4BHK',
        '2 bhk': '2BHK', '2 bedroom villas': 'Villa', '2 bhk;2bhk': '2BHK',
        '2 bhk;3 bhk 3t': '3BHK', '3 bedroom villas': 'Villa', '4 bedroom villa': 'Villa',
        '2 bedroom villa': 'Villa', '5 bedroom villa': 'Villa', '1 bhk': '1BHK',
        '2 bhk;plots': 'Plots', '3 bedroom villa': 'Villa', '2 bhk;3 bhk 2t': '2BHK',
        '3 bhk 2t;4 bhk': '3BHK', '4 bedroom villas': 'Villa',
        '3 bhk 2t;3 bhk 3t;4 bhk': '3BHK', '2 bhk 2t': '2BHK'
    }
    s_pt = df_merged['Preferred Property Type'].astype(str).str.lower().str.strip()
    s_pt = s_pt.str.replace(r'\s+', ' ', regex=True).str.replace(r'\s*;\s*', ';', regex=True)
    map_pt_nospace = {k.replace(' ', ''): v for k, v in map_pt.items()}
    mapped_pt = s_pt.map(map_pt)
    fallback_pt = s_pt.str.replace(' ', '', regex=False).map(map_pt_nospace)
    mapped_pt = mapped_pt.mask(mapped_pt.isna(), fallback_pt)   
    df_merged['Preferred Property Type'] = mapped_pt.fillna('Others')

    # SFT range normalization (vectorized)
    s_sft = df_merged['SFT Range'].astype(str).str.strip().str.lower()
    conds = [
        (s_sft == '<500'),
        (s_sft == '> 5001'),
        s_sft.str.contains('501 - 1000', regex=False),
        s_sft.str.contains('1001 - 1250', regex=False),
        s_sft.str.contains('1251 - 1500', regex=False),
        s_sft.str.contains('1501 - 1750', regex=False),
        s_sft.str.contains('1751 - 2000', regex=False),
        s_sft.str.contains('2001 - 2250', regex=False),
        s_sft.str.contains('2251 - 2500', regex=False),
        s_sft.str.contains('2501 - 2750', regex=False),
        s_sft.str.contains('2751 - 3000', regex=False),
        s_sft.str.contains('3001 - 3250', regex=False),
        s_sft.str.contains('3251 - 3500', regex=False),
        s_sft.str.contains('3501 - 3750', regex=False),
        s_sft.str.contains('3751 - 4000', regex=False),
        s_sft.str.contains('4001 - 4250', regex=False),
        s_sft.str.contains('4251 - 4500', regex=False),
        s_sft.str.contains('4501 - 4750', regex=False),
        s_sft.str.contains('4751 - 5000', regex=False)
    ]
    choices = [
        '< 500', '> 5001', '501 - 1000', '1001 - 1250', '1251 - 1500',
        '1501 - 1750', '1751 - 2000', '2001 - 2250', '2251 - 2500', '2501 - 2750',
        '2751 - 3000', '3001 - 3250', '3251 - 3500', '3501 - 3750', '3751 - 4000',
        '4001 - 4250', '4251 - 4500', '4501 - 4750', '4751 - 5000'
    ]
    df_merged['SFT Range'] = np.select(conds, choices, default='Others')

    # Kids normalization (vectorized)
    df_merged['How Many Kids?'] = (
    df_merged['How Many Kids?']
    .astype(str)                # convert everything to string for consistent mapping
    .str.strip()               # remove accidental spaces
    .replace({
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '3 or more': 3  # OR keep as string if needed → change below
    })
    .astype('Int64'))

    # Income mapping (vectorized numeric-first, then string fallback)
    income_col = 'Household Monthly Income (Rs)'
    categories_income = [
        'Rs. 50,000 or less',
        'Rs. 50,000 - Rs. 1 Lakh',
        'Rs. 1 Lakh to Rs. 2 Lakhs',
        'Rs. 2 Lakhs to Rs. 3 Lakhs',
        'Rs. 3 Lakhs to Rs. 4 Lakhs',
        'Rs. 4 Lakhs to Rs. 5 Lakhs',
        'Rs. 5 Lakhs to Rs. 7.5 Lakhs',
        'Rs. 7.5 Lakhs to Rs. 10 Lakhs',
        'Rs. 10 Lakhs to Rs. 12.5 Lakhs',
        'Rs. 12.5 Lakhs to Rs. 15 Lakhs',
        'Rs. 15 Lakhs and above'
    ]
    bins_income = [-np.inf, 50000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000, 1250000, 1500000, np.inf]

    income_series = df_merged[income_col]
    income_num = pd.to_numeric(income_series, errors='coerce')
    income_bucket_num = pd.cut(income_num, bins=bins_income, labels=categories_income, right=True)

    # String fallback parsing
    s_income = income_series.astype(str).str.strip().str.replace('Laksh', 'Lakhs', regex=False)
    is_known = s_income.isin(categories_income)
    income_bucket_str_known = s_income.where(is_known)

    # Extract first numeric in string where not known
    s_num = pd.to_numeric(
    s_income.where(~is_known)
    .str.replace(',', '', regex=False)
    .str.extract(r'([\d.]+)', expand=False),  # raw string; '.' need not be escaped inside [...]
    errors='coerce')
    income_bucket_str_num = pd.cut(s_num, bins=bins_income, labels=categories_income, right=True)

    # Combine buckets (priority: numeric -> known string -> extracted numeric)
    income_bucket = income_bucket_num.astype(object)
    income_bucket = income_bucket.fillna(income_bucket_str_known)
    income_bucket = income_bucket.fillna(income_bucket_str_num)
    df_merged[income_col] = income_bucket.fillna('Income range not recognized')

    # Binning Outgoing calls (vectorized)
    df_merged['Total Outgoing Answered Calls'] = pd.to_numeric(df_merged['Total Outgoing Answered Calls'], errors='coerce')
    df_merged['Total Outgoing Answered Calls'] = pd.cut(
        df_merged['Total Outgoing Answered Calls'],
        bins=[-np.inf, 2, np.inf],
        labels=['0-2', '3 and above']
    )

    # Binning Incoming calls (vectorized)
    df_merged['Total Incoming Answered Calls'] = pd.to_numeric(df_merged['Total Incoming Answered Calls'], errors='coerce')
    df_merged['Total Incoming Answered Calls'] = pd.cut(
        df_merged['Total Incoming Answered Calls'],
        bins=[-np.inf, 1, np.inf],
        labels=['0-1', '2 and above']
    )

    # Age bins (vectorized)
    df_merged['Age'] = pd.to_numeric(df_merged['Age'], errors='coerce')
    df_merged['Age'] = pd.cut(
        df_merged['Age'],
        bins=[-np.inf, 37, 42, 49, np.inf],
        labels=['18_to_37', '38_to_42', '43_to_49', '50_and_above']
    )

    #Re-engagement count bins (vectorized)
    rec = pd.to_numeric(df_merged['Re-engagement count'], errors='coerce')
    df_merged['Re-engagement count'] = np.select(
        [
            rec >= 2,
            rec >= 0
        ],
        [
            'Grtr_thn_2',
            '0_to_2'
        ],
        default='Other'
    )

    # Developers bin (vectorized)
    devs = pd.to_numeric(df_merged['DEVELOPERS YOU WORK WITH_STATUS_1'], errors='coerce')
    df_merged['DEVELOPERS YOU WORK WITH_STATUS_1'] = np.where(devs <= 9, '0-9', 'Above_9')

    # Strong micro market bin (vectorized)
    smm = pd.to_numeric(df_merged['WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1'], errors='coerce')
    df_merged['WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1'] = np.where(smm <= 3, '0to3', '4 and above')

    # Other markets bin (vectorized)
    om = pd.to_numeric(df_merged['WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1'], errors='coerce')
    df_merged['WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1'] = np.where(om <= 2, '0-2', '3 and above')

    # Team size bin (vectorized)
    ts = pd.to_numeric(df_merged['TEAM SIZE_STATUS'], errors='coerce')
    df_merged['TEAM SIZE_STATUS'] = np.where(ts <= 10, 'less_than_eqlto_10', '11 and above')

    # # Quotation bins (vectorized)
    # qnum = pd.to_numeric(df_merged['Quotation'], errors='coerce').fillna(-1)
    # df_merged['Quotation'] = pd.cut(
    #     qnum,
    #     bins=[-np.inf, 0, 2, 10, np.inf],
    #     labels=['Q0', 'Q1-2', 'Q3-10', 'Q11+'],
    #     right=True
    # )

    # Quotation bins (vectorized)
    df_merged['Quotation'] = np.where(df_merged['Quotation']>0,1,0)
    #print("third check",df_merged.shape)
    
    # Unsold inventory bins for percentage (vectorized)
    uip = pd.to_numeric(df_merged["Sum of Unsold inventory"], errors='coerce')
    df_merged["Sum of Unsold inventory"] = pd.cut(
        uip,
        bins=[-np.inf, 15.85, 32.68, 100, np.inf],
        labels=["lessthan15.85per", "15.86perto32.68per", "32.69perto100perc", "Unknown"],
        right=True
    ).astype(object)

    # Kids grouping on base df (vectorized)
    def categorize_kids(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in ['0', '1']:
            return '0-1'
        elif x in ['2', '3', '3 or more']:
            return '>=2'
        else:
            return np.nan
    
    df_merged['How Many Kids?'] = df_merged['How Many Kids?'].apply(categorize_kids).astype('object')


    # Household Monthly Income grouped (vectorized map)
    income_group_map = {
        'Rs. 50,000 or less': 'Rs. 50,000 or less',
        'Rs. 50,000 - Rs. 1 Lakh': 'Rs. 50,000 - Rs. 1 Lakh',
        'Rs. 1 Lakh to Rs. 2 Lakhs': 'Rs. 1 Lakh to Rs. 4 Lakhs',
        'Rs. 2 Lakhs to Rs. 3 Lakhs': 'Rs. 1 Lakh to Rs. 4 Lakhs',
        'Rs. 3 Lakhs to Rs. 4 Lakhs': 'Rs. 1 Lakh to Rs. 4 Lakhs',
        'Rs. 4 Lakhs to Rs. 5 Lakhs': 'Rs. 4 Lakhs to Rs. 10 Lakhs',
        'Rs. 5 Lakhs to Rs. 7.5 Lakhs': 'Rs. 4 Lakhs to Rs. 10 Lakhs',
        'Rs. 7.5 Lakhs to Rs. 10 Lakhs': 'Rs. 4 Lakhs to Rs. 10 Lakhs',
        'Rs. 10 Lakhs to Rs. 12.5 Lakhs': 'Above Rs. 10 Lakhs',
        'Rs. 12.5 Lakhs to Rs. 15 Lakhs': 'Above Rs. 10 Lakhs',
        'Rs. 15 Lakhs and above': 'Above Rs. 10 Lakhs',
        'Income range not recognized': 'Income range not recognized'
    }
    df_merged['Household Monthly Income (Rs)'] = df_merged['Household Monthly Income (Rs)'].map(income_group_map).fillna('Income range not recognized')

    # Budget grouping (vectorized map)
    budget_group_map = {
        'Less than 50 Lakhs': 'Less than 50 Lakhs',
        '50 Lakhs - 75 Lakhs': '50 Lakhs - 75 Lakhs',
        '75 Lakhs - 1 Crore': '75 Lakhs - 1 Crore',
        '1 Crore - 1.5 Crores': '1 Crore - 2 Crores',
        '1.5 Crores - 2 Crores': '1 Crore - 2 Crores',
        '1 - 1.2 Crore': '1 Crore - 2 Crores',
        '1.8 - 2 Crore': '1 Crore - 2 Crores',
        '1 Crore - 2 Crores': '1 Crore - 2 Crores',
        '2 Crores - 3 Crores': 'Above Rs. 2 Crores',
        '3 Crores - 4 Crores': 'Above Rs. 2 Crores',
        '4 Crores - 5 Crores': 'Above Rs. 2 Crores',
        '5 Crores - 7.5 Crores': 'Above Rs. 2 Crores',
        '7.5 Crores - 10 Crores': 'Above Rs. 2 Crores',
        '10 Crores - 15 Crores': 'Above Rs. 2 Crores',
        '15 Crores and above': 'Above Rs. 2 Crores'
    }
    df_merged['Budget'] = df_merged['Budget'].map(budget_group_map).fillna('Budget range not recognized')

    # Interested in normalization -> simplified (vectorized map)
    interest_map = {
        'Plots': 'Plots',
        'Apartments': 'Apartments',
        'Villas': 'Villa/Row Houses',
        'Villa': 'Villa/Row Houses',
        'Row House': 'Villa/Row Houses',
        'Villas / Row Houses': 'Villa/Row Houses',
        '1BHK': '1BHK',
        '1 BHK': '1BHK',
        '2BHK': '2BHK',
        '2 BHK': '2BHK',
        '3BHK': '3BHK',
        '3 BHK': '3BHK',
        '4BHK': '4BHK',
        '4 BHK': '4BHK'
    }
    if 'Interested in' in df_merged.columns:
        df_merged['Interested in'] = df_merged['Interested in'].map(interest_map).fillna('Interest type not recognized')
        df_merged['Interested in'] = df_merged['Interested in'].replace('Villas', 'Villa/Row Houses')

    # SFT Range grouping (vectorized)
    sft_group_map = {
        '< 500': '<500',
        '501 - 1000': '501 - 1000',
        '1001 - 1250': '1001 - 1250',
        '1251 - 1500': '1251 - 1500'
    }
    s = df_merged['SFT Range'].map(sft_group_map)
    
    fallback_arr = np.where(
        df_merged['SFT Range'].isin([
            '1501 - 1750', '1751 - 2000', '1751 - 2250', '2001 - 2250', '2251 - 2500',
            '2501 - 2750', '2751 - 3000', '3001 - 3250', '3251 - 3500', '3501 - 3750',
            '3751 - 4000', '4001 - 4250', '4251 - 4500', '4501 - 4750', '4751 - 5000', '> 5001'
        ]),
        '>1500',
        'Sqft range not recognized'
    )

    fallback_ser = pd.Series(fallback_arr, index=df_merged.index)
    df_merged['SFT Range'] = s.fillna(fallback_ser)

    # Target sale range (vectorized)
    tsale = pd.to_numeric(df_merged['TARGET SALE FOR THIS FINANCIAL YEAR_STATUS'], errors='coerce')
    df_merged['TARGET SALE FOR THIS FINANCIAL YEAR_STATUS'] = np.select(
        [
            (tsale >= 10) & (tsale < 40),
            (tsale >= 40) & (tsale <= 500)
        ],
        [
            'above10-blw40',
            '40-500'
        ],
        default='Out of Range'
    )

    # Keep simplified interested in final form
    if 'Interested in' in df_merged.columns:
        df_merged['Interested in'] = df_merged['Interested in'].map({
            'Plots': 'Plots',
            'Apartments': 'Apartments',
            'Villa/Row Houses': 'Villa/Row Houses'
        }).fillna('Interest type not recognized')

    # Create df_1 with reduced columns
    df_1 = df_merged.drop(
        columns=['Applicant - Nature of Occupation', 'Total Incoming Answered Calls',
                 'Total Outgoing Answered Calls', 'AI_hour', 'AI/ML Object.1'],
        errors='ignore'
    )

    # Appointment month start
    df_1['Appointment_month_start'] = df_1['Appointment_Initiated_Date'].apply(
        lambda x: pd.Timestamp(year=x.year, month=x.month, day=1) if pd.notnull(x) else pd.NaT
    )

    # Merge with df_cyclic (typed keys, no string concatenation)
    df_cyclic = df_cyclic.copy()
    df_cyclic['Calculated For SV_month_year'] = pd.to_datetime(
        df_cyclic['Calculated For SV_month_year'], format="%Y-%m-%d", errors='coerce'
    )
    df_cyclic["next_month"] = (df_cyclic["Calculated For SV_month_year"] + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()

    df_cyclic['CP_Name_clean'] = df_cyclic['CP_Name'].apply(clean_text)
    df_1['CP_Name_clean'] = df_1['CP_Name'].apply(clean_text)

    df_merged = pd.merge(
        df_1,
        df_cyclic.drop(columns=['CP_Name'], errors='ignore'),
        left_on=['CP_Name_clean', 'Appointment_month_start'],
        right_on=['CP_Name_clean', 'next_month'],
        how='left',
        sort=False
    )

    # Replacing NaN values with 0 for specified columns
    columns_to_fill = [
        '1to1', '1to2', '1to3', '1to4', '1to5', '1to6', '1to7',
        '1to8', '1to9', '1to10', '1to11', '1to12', '1to13',
        '1to14', '1to15', '1to16', '1to17', '1to18'
    ]
    existing_fill_cols = [c for c in columns_to_fill if c in df_merged.columns]
    if existing_fill_cols:
        df_merged[existing_fill_cols] = df_merged[existing_fill_cols].fillna(0)

    # Interaction Variable
    if set(['Reason for Purchase', 'Is this your First Home Purchase?']).issubset(df_merged.columns):
        df_merged['Purchasereason_Firsthome_Interaction'] = (
            df_merged['Reason for Purchase'].astype(str) + '_' + df_merged['Is this your First Home Purchase?'].astype(str)
        )
        df_merged = df_merged.drop(['Reason for Purchase', 'Is this your First Home Purchase?'], axis=1)

    # marital status_kids_strength
    if set(['Marital Status', 'How Many Kids?']).issubset(df_merged.columns):
        df_merged['Maritial_Kids_Interaction'] = (
            df_merged['Marital Status'].astype(str) + '_' + df_merged['How Many Kids?'].astype(str)
        )
        df_merged = df_merged.drop(['Marital Status', 'How Many Kids?'], axis=1)

        mapping = {
            'Married_1': 'Married with children_1',
            'Married_2': 'Married with children_2',
            'Married_3': 'Married with children_3',
            'Married_3 or more': 'Married with children_3 or more',
            'Single_0': 'Single',
            'Single_1': 'Single',
            'Single_2': 'Single'
        }
        df_merged['Maritial_Kids_Interaction'] = df_merged['Maritial_Kids_Interaction'].replace(mapping)

    # saleincentive_club
    if set(['INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS', 'Club Type_STATUS']).issubset(df_merged.columns):
        df_merged['saleincentive_club'] = (
            df_merged['INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS'].astype(str)
            + '_' +
            df_merged['Club Type_STATUS'].astype(str)
        )
        df_merged = df_merged.drop(['INTERESTED IN ANNUAL INCENTIVE PROGRAM_STATUS', 'Club Type_STATUS'], axis=1)

    # Drop date helpers
    df_merged = df_merged.drop(['Calculated For SV_month_year', 'Appointment_month_start', 'next_month'], axis=1, errors='ignore')

    # Drop specific columns
    columns_to_drop = ['SV_Duration', '1t013', '1to14', '1to15', '1to16', '1to17', '1to18']
    df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

    # Convert float columns to int where present
    float_columns = [
        '1to1', '1to2', '1to3', '1to4', '1to5', '1to6', '1to7',
        '1to8', '1to9', '1to10', '1to11', '1to12', '1to13'
    ]
    existing_float_cols = [c for c in float_columns if c in df_merged.columns]
    for c in existing_float_cols:
        df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(0).astype(int)

    # Remove specified columns
    columns_to_remove = [
        'Interested in',
        'ACCOUNT TYPE_STATUS',
        'saleincentive_club',
        'project_clean',
        'month_start'  # helper from early join
    ]
    df_merged = df_merged.drop(columns=columns_to_remove, errors='ignore')

    # Final cleanup of original date/id columns (adjusted to current keys)
    df_merged = df_merged.drop(
        columns=['Project Launch Date', 'concat', 'Formatted Date', 'Applicant - Industry'],
        errors='ignore'
    )

    # Convert frequent low-cardinality strings to categorical dtype to reduce memory / speed ops
    categorical_candidates = [
        'Gender', 'Current Accomodation Type', 'Household Monthly Income (Rs)', 'Budget',
        'Refer a family or friend', 'Preferred Property Type', 'SFT Range', 'Category',
        'Quotation', 'Sum of Unsold inventory', 'Launch_Sustenance', 'Binned_Launch_to_Appointment',
        'AI_day', 'CP Type', 'DEVELOPERS YOU WORK WITH_STATUS_1',
        'WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1',
        'WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1', 'TEAM SIZE_STATUS',
        'TARGET SALE FOR THIS FINANCIAL YEAR_STATUS', 'Purchasereason_Firsthome_Interaction',
        'Maritial_Kids_Interaction','Re-engagement count', 'Appointment_Initiated_Date','Project_Interested'
    ]
    for col in categorical_candidates:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype('category')
     
    # Cleaned, disjoint keyword groups
    group_keywords = {
        "Time to think": [
            "need some time", "give time", "will respond by", "respond by", "need time",
            "require time", "more time", "later", "delay", "follow up later", "not now", "will get back", "call back"
        ],
        "Future visit": [
            "will visit", "visit again", "planning visit", "with family", "tomorrow",
            "saturday", "sunday", "today", "am", "pm", "2mrw", "weekend", "this week", "next week"
        ],
        "Virtual meet": [
            "virtual meet", "virtual", "vc", "video call", "zoom", "teams", "online call", "google meet"
        ],
        "Purchase intent": [
            "eoi", "expression of interest", "cheque", "token amount", "will pay",
            "payment intent", "initiate payment"
        ],
        "Project details": [
            "shared project details", "sent on whatsapp", "shared on whatsapp", "mail",
            "emailed", "project info", "brochure", "specifications", "master plan", "location map"
        ],
        "Payment plans or cost sheet": [
            "cost sheet", "payment plans", "balance payment", "payment breakup",
            "pricing sheet", "stage wise payment", "payment schedule", "emi"
        ],
        "Type of purchase": [
            "investment purpose", "investment", "end use", "own use", "self use",
            "for rental", "second home"
        ],
        "Within_Budget": [
            "within budget", "in budget", "cr", "crore", "lakhs", "budget is",
            "price bracket", "price range", "cost is"
        ],
        "Decision maker": [
            "wife", "mother", "father", "spouse", "family will decide", "parents",
            "discuss with family", "decision pending", "brother in law", "partner", "sibling"
        ],
        "Flat preference": [
            "selected unit", "interested in unit", "looking for unit", "booked",
            "blocking", "blocked", "shortlisted", "formalities", "deal", "finalizing", "finalized"
        ],
        "Facing preference": [
            "north facing", "south facing", "east facing", "west facing",
            "north", "south", "east", "west", "corner", "vaastu facing"
        ],
        "Not interested": [
            "rnr", "not reachable", "not interested", "no response", "no call back",
            "not picking", "declined", "not pursuing", "dropped out", "ignoring", "moved on", "busy", "not responding", "dropped"
        ],
        "Seeking property type": [
            "looking for plots", "looking for villa", "villa buyer", "flat buyer",
            "prefer plot", "plots only", "villa preferred", "apartment seeker", "2bhk", "3bhk"
        ],
        "Budget constraints": [
            "exceeding budget", "out of budget", "pricing issue", "price too high",
            "can’t afford", "budget mismatch", "beyond budget", "not matching budget", "expectations not met"
        ],
        "Vaastu constraints": [
            "vastu", "vaastu compliant", "not vastu compliant", "not matching vaastu", "not aligned with vaastu"
        ],
        "Amenities constraints": [
            "basketball", "tennis", "shuttle cock", "badminton", "gym", "swimming pool",
            "clubhouse missing", "lack of facilities", "no amenities", "no play area", "fewer amenities"
        ],
        "Property constraints": [
            "too small", "not enough space", "need more space", "roof height", "balcony small",
            "bad dimensions", "size issue", "floor preference", "no sunlight", "too low",
            "not enough ventilation", "bad layout"
        ]
    }

    # Create binary flags for each group
    for group, keywords in group_keywords.items():
        pattern = '|'.join([rf'\b{k}\b' if ' ' not in k else k for k in [kw.lower() for kw in keywords]])
        df_merged[group] = df_merged['Call_remarks'].str.contains(pattern, case=False, na=False).astype(int)

    df_merged["Type_of_SV"] = df_merged["Launch_Sustenance"]
    
    # Ensure required output columns exist
    required_columns = ['Enquiry_Ref_No.',
        'Gender', 'Age', 'Current Accomodation Type',
        'Household Monthly Income (Rs)', 'Budget', 'Refer a family or friend',
        'Re-engagement count', 'Preferred Property Type', 'SFT Range',
        'Category', 'Quotation', 'Sum of Unsold inventory',
        'Binned_Launch_to_Appointment', 'AI_day', 'CP Type',
        'DEVELOPERS YOU WORK WITH_STATUS_1',
        'WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1',
        'WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1', 'TEAM SIZE_STATUS',
        'TARGET SALE FOR THIS FINANCIAL YEAR_STATUS', '1to1', '1to2', '1to3',
        '1to4', '1to5', '1to6', '1to7', '1to8', '1to9', '1to10', '1to11',
        '1to12', '1to13', 'Purchasereason_Firsthome_Interaction',
        'Maritial_Kids_Interaction','Appointment_Initiated_Date','Project_Interested','Time to think','Future visit',
        'Virtual meet','Purchase intent','Project details',
        'Payment plans or cost sheet','Type of purchase','Within_Budget',
        'Decision maker','Flat preference','Facing preference','Not interested',
        'Seeking property type','Budget constraints','Vaastu constraints',
        'Amenities constraints','Property constraints', 'Type_of_SV'
    ]
    for col in required_columns:
        if col not in df_merged.columns:
            df_merged[col] = np.nan

    df_final = df_merged[required_columns].copy()
    
    
    #Dummy encoding
    columns_to_encode =['Gender','Age','Current Accomodation Type','Household Monthly Income (Rs)','Budget','Refer a family or friend','Re-engagement count','Preferred Property Type','SFT Range','Category','Sum of Unsold inventory','Binned_Launch_to_Appointment','AI_day','CP Type','DEVELOPERS YOU WORK WITH_STATUS_1','WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1','WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1','TEAM SIZE_STATUS','TARGET SALE FOR THIS FINANCIAL YEAR_STATUS','Maritial_Kids_Interaction','Type_of_SV']

    df_encoded = pd.get_dummies(df_final, columns=columns_to_encode)

    df_encoded = df_encoded.drop(columns=['Appointment_Initiated_Date','Project_Interested'])
    
    # Boolean columns remain unchanged (but fill NaN as False)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].fillna(False)
    df_encoded[bool_cols]=df_encoded[bool_cols].astype(int)
    
    #Columns to be present , if not create and put it as zeroes
    Required_cols = ['1to1','1to2','1to3','1to4','1to5','1to6','1to7','1to8','1to9','1to10','1to11','1to12','1to13'
                     ,'Time to think','Future visit','Virtual meet','Purchase intent','Project details',
                    'Payment plans or cost sheet','Type of purchase','Within_Budget','Decision maker','Flat preference',
                    'Facing preference','Not interested','Seeking property type','Budget constraints','Vaastu constraints',
                    'Amenities constraints','Property constraints','Gender_Male','Age_38_to_42','Age_43_to_49',
                    'Age_50_and_above','Current Accomodation Type_company-provided-house','Current Accomodation Type_rent-house'
                     ,'Current Accomodation Type_self-own-house','Household Monthly Income (Rs)_Rs. 1 Lakh to Rs. 4 Lakhs',
                    'Household Monthly Income (Rs)_Rs. 4 Lakhs to Rs. 10 Lakhs','Household Monthly Income (Rs)_Rs. 50,000 - Rs. 1 Lakh',
                    'Household Monthly Income (Rs)_Rs. 50,000 or less','Budget_50 Lakhs - 75 Lakhs','Budget_75 Lakhs - 1 Crore',
                    'Budget_Above Rs. 2 Crores','Budget_Less than 50 Lakhs','Refer a family or friend_Yes','Re-engagement count_Grtr_thn_2',
                    'Preferred Property Type_2BHK','Preferred Property Type_3BHK','Preferred Property Type_4BHK',
                    'Preferred Property Type_Others','Preferred Property Type_Plots','Preferred Property Type_Villa',
                    'SFT Range_1251 - 1500','SFT Range_501 - 1000','SFT Range_<500','SFT Range_>1500','Category_NRI',
                    'Sum of Unsold inventory_32.69perto100perc','Sum of Unsold inventory_lessthan15.85per',
                    'Binned_Launch_to_Appointment_-180 to -91','Binned_Launch_to_Appointment_-91 to -61','Binned_Launch_to_Appointment_-61 to -31',
                    'Binned_Launch_to_Appointment_-31 to -1','Binned_Launch_to_Appointment_0-30','Binned_Launch_to_Appointment_31-60',
                    'Binned_Launch_to_Appointment_61-90','Binned_Launch_to_Appointment_91-180','Binned_Launch_to_Appointment_180+',
                    'AI_day_Monday','AI_day_Saturday','AI_day_Sunday','AI_day_Thursday','AI_day_Tuesday','AI_day_Wednesday',
                    'CP Type_NRI-CP','DEVELOPERS YOU WORK WITH_STATUS_1_Above_9','WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1_4 and above',
                    'WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1_3 and above','TEAM SIZE_STATUS_less_than_eqlto_10',
                    'TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_Out of Range','TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_above10-blw40',
                    'Maritial_Kids_Interaction_Married with children_>=2','Maritial_Kids_Interaction_Married with children_nan',
                    'Maritial_Kids_Interaction_Married_0-1','Maritial_Kids_Interaction_Married_>=2','Maritial_Kids_Interaction_Married_nan',
                    'Maritial_Kids_Interaction_Single_0-1','Maritial_Kids_Interaction_Single_>=2','Maritial_Kids_Interaction_Single_nan',
                    'Type_of_SV_sustenance']
    
    df_encoded = df_encoded.reindex(columns=Required_cols, fill_value=0)

    return df_encoded
   # return df_merged

def CP_model_predict1(df):
    CPsales_LS_model = joblib.load(r'LS_Event1_withoutQ.pkl')
    CPsales_pred = CPsales_LS_model.predict_proba(df)[:, 1]
    classification = 'High Likely to Buy' if CPsales_pred >= 0.155833 else 'Less likely to Buy'
    return CPsales_pred, classification

def cp_shap_scores(df):

    import joblib
    import pandas as pd

    # ---------------------------
    # 1. Feature Mapping Dictionary (unchanged)
    # ---------------------------
    feature_map = {      
        # Continuous SV months (1to1 to 1to13)
        "1to1": "Months CP continuously gave SVs 1",
        "1to2": "Months CP continuously gave SVs 2",
        "1to3": "Months CP continuously gave SVs 3",
        "1to4": "Months CP continuously gave SVs 4",
        "1to5": "Months CP continuously gave SVs 5",
        "1to6": "Months CP continuously gave SVs 6",
        "1to7": "Months CP continuously gave SVs 7",
        "1to8": "Months CP continuously gave SVs 8",
        "1to9": "Months CP continuously gave SVs 9",
        "1to10": "Months CP continuously gave SVs 10",
        "1to11": "Months CP continuously gave SVs 11",
        "1to12": "Months CP continuously gave SVs 12",
        "1to13": "Months CP continuously gave SVs 13",

        # Customer interest/behavior
        "Time to think": "Customer wants time to think",
        "Future visit": "Interested in future visit",
        "Virtual meet": "Interested in virtual meeting",
        "Purchase intent": "Purchase intent expressed",
        "Project details": "Asked for project details",
        "Payment plans or cost sheet": "Asked for pricing/payment plans",
        "Type of purchase": "Type of purchase discussed",
        "Within_Budget": "Within budget",
        "Decision maker": "Decision maker identified",
        "Flat preference": "Flat preference specified",
        "Facing preference": "Facing preference specified",
        "Not interested": "Not interested",
        "Seeking property type": "Property type requirement",
        "Budget constraints": "Budget related concern",
        "Vaastu constraints": "Vaastu preference constraint",
        "Amenities constraints": "Amenities concern",
        "Property constraints": "Property related constraint",

        # Demographics
        "Gender_Female": "Female",
        "Gender_Male": "Male",
        "Age_18_to_37": "Age: 18 to 37",
        "Age_38_to_42": "Age: 38 to 42",
        "Age_43_to_49": "Age: 43 to 49",
        "Age_50_and_above": "Age: 50+",

        # Current accommodation
        "Current Accomodation Type_Rented": "Rented house",
        "Current Accomodation Type_company-provided-house": "Company-provided house",
        "Current Accomodation Type_rent-house": "Rented house",
        "Current Accomodation Type_self-own-house": "Own house",

        # Income
        "Household Monthly Income (Rs)_Above Rs. 10 Lakhs": "Income: more than 10L",
        "Household Monthly Income (Rs)_Rs. 1 Lakh to Rs. 4 Lakhs": "Income: 1L to 4L",
        "Household Monthly Income (Rs)_Rs. 4 Lakhs to Rs. 10 Lakhs": "Income: 4L to 10L",
        "Household Monthly Income (Rs)_Rs. 50,000 - Rs. 1 Lakh": "Income: 50K to 1L",
        "Household Monthly Income (Rs)_Rs. 50,000 or less": "Income: ≤50K",

        # Budget
        "Budget_1 Crore - 2 Crores": "Budget: 1Cr to 2Cr",
        "Budget_50 Lakhs - 75 Lakhs": "Budget: 50 to 75L",
        "Budget_75 Lakhs - 1 Crore": "Budget: 75L to 1Cr",
        "Budget_Above Rs. 2 Crores": "Budget: >2Cr",
        "Budget_Less than 50 Lakhs": "Budget: <50L",

        # Referrals & Engagement
        "Refer a family or friend_No": "No Referral given",
        "Refer a family or friend_Yes": "Referral given",
        "Re-engagement count_0_to_2": "Less re-engagement",
        "Re-engagement count_Grtr_thn_2": "High re-engagement (>2 times)",

        # Property preferences
        "Preferred Property Type_1BHK": "Preference: 1 BHK",
        "Preferred Property Type_2BHK": "Preference: 2 BHK",
        "Preferred Property Type_3BHK": "Preference: 3 BHK",
        "Preferred Property Type_4BHK": "Preference: 4 BHK",
        "Preferred Property Type_Others": "Other property type",
        "Preferred Property Type_Plots": "Preference: Plots",
        "Preferred Property Type_Villa": "Preference: Villa",

        # Size preferences
        "SFT Range_1001 - 1250": "Size: 1001 to 1250 sqft",
        "SFT Range_1251 - 1500": "Size: 1251 to 1500 sqft",
        "SFT Range_501 - 1000": "Size: 501 to 1000 sqft",
        "SFT Range_<500": "Size: <500 sqft",
        "SFT Range_>1500": "Size: >1500 sqft",

        # Customer category
        "Category_Indian": "Indian customer",
        "Category_NRI": "NRI customer",

        # Inventory
        "Sum of Unsold inventory_15.86perto32.68per": "Med unsold inventory",
        "Sum of Unsold inventory_32.69perto100perc": "High unsold inventory",
        "Sum of Unsold inventory_lessthan15.85per": "Low unsold inventory",

        # Launch timing
        "Binned_Launch_to_Appointment_-180+": "Very early pre-launch engagement",
        "Binned_Launch_to_Appointment_-180 to -91": "Very early pre-launch engagement",
        "Binned_Launch_to_Appointment_-91 to -61": "Early pre-launch engagement",
        "Binned_Launch_to_Appointment_-61 to -31": "Mid pre-launch engagement",
        "Binned_Launch_to_Appointment_-31 to -1": "Late pre-launch engagement",
        "Binned_Launch_to_Appointment_0-30": "0 to 30 days post launch",
        "Binned_Launch_to_Appointment_31-60": "31 to 60 days post launch",
        "Binned_Launch_to_Appointment_61-90": "61 to 90 days post launch",
        "Binned_Launch_to_Appointment_91-180": "91 to 180 days post launch",
        "Binned_Launch_to_Appointment_180+": ">180 days post launch",

        # Interaction day
        "AI_day_Friday": "Interaction on Friday",
        "AI_day_Monday": "Interaction on Monday",
        "AI_day_Saturday": "Interaction on Saturday",
        "AI_day_Sunday": "Interaction on Sunday",
        "AI_day_Thursday": "Interaction on Thursday",
        "AI_day_Tuesday": "Interaction on Tuesday",
        "AI_day_Wednesday": "Interaction on Wednesday",

        # CP Type
        "CP Type_Domestic-CP": "Channel Partner: Domestic",
        "CP Type_NRI-CP": "Channel Partner: NRI",

        # CP Developer/Market info
        "DEVELOPERS YOU WORK WITH_STATUS_1_0-9": "Works with many developers (<9)",
        "DEVELOPERS YOU WORK WITH_STATUS_1_Above_9": "Works with many developers (>9)",
        "WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1_0to3": "Strong in multiple micro markets (<4)",
        "WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1_4 and above": "Strong in multiple micro markets (>4)",
        "WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1_0-2": "Operates in multiple markets (<3)",
        "WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1_3 and above": "Operates in multiple markets (>3)",

        # CP Team & Targets
        "TEAM SIZE_STATUS_11 and above": "CP with big team size",
        "TEAM SIZE_STATUS_less_than_eqlto_10": "CP with small team size",
        "TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_40-500": "Sales target: More than 40",
        "TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_Out of Range": "Sales target not defined",
        "TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_above10-blw40": "Sales target: 10 to 40",

        # Marital & Kids
        "Maritial_Kids_Interaction_Married with children_0-1": "Married with 1 child",
        "Maritial_Kids_Interaction_Married with children_>=2": "Married with 2+ children",
        "Maritial_Kids_Interaction_Married with children_nan": "Married, children not disclosed",
        "Maritial_Kids_Interaction_Married_0-1": "Married with ≤1 child",
        "Maritial_Kids_Interaction_Married_>=2": "Married with ≥2 children",
        "Maritial_Kids_Interaction_Married_nan": "Married, details missing",
        "Maritial_Kids_Interaction_Single_0-1": "Single with ≤1 child",
        "Maritial_Kids_Interaction_Single_>=2": "Single with ≥2 children",
        "Maritial_Kids_Interaction_Single_nan": "Single, details missing",

        # Site visit type
        "Type_of_SV_launch": "Site visited during launch phase",
        "Type_of_SV_sustenance": "Site visited during sustenance phase"
    }

    # ---------------------------
    # 2. Feature Control Dictionary (FROM YOUR TABLE)
    # ---------------------------
    feature_control = {

        # Hidden features
        **{f"1to{i}": {"visible": "No", "driver": ""} for i in range(1, 14)},
        "Gender_Female": {"visible": "No", "driver": ""},
        "Gender_Male": {"visible": "No", "driver": ""},
        "Refer a family or friend_No": {"visible": "No", "driver": ""},
        "Refer a family or friend_Yes": {"visible": "No", "driver": ""},
        "Category_Indian": {"visible": "No", "driver": ""},
        "CP Type_Domestic-CP": {"visible": "No", "driver": ""},
        "CP Type_NRI-CP": {"visible": "No", "driver": ""},
        "Type_of_SV_launch": {"visible": "No", "driver": ""},
        "Type_of_SV_sustenance": {"visible": "No", "driver": ""},

        # Positive only
        "Household Monthly Income (Rs)_Above Rs. 10 Lakhs": {"visible": "Yes", "driver": "Positive"},
        "Household Monthly Income (Rs)_Rs. 1 Lakh to Rs. 4 Lakhs": {"visible": "Yes", "driver": "Positive"},
        "Household Monthly Income (Rs)_Rs. 4 Lakhs to Rs. 10 Lakhs": {"visible": "Yes", "driver": "Positive"},
        "Re-engagement count_Grtr_thn_2": {"visible": "Yes", "driver": "Positive"},
        "Sum of Unsold inventory_32.69perto100perc": {"visible": "Yes", "driver": "Positive"},
        "WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1_4 and above": {"visible": "Yes", "driver": "Positive"},
        "TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_40-500": {"visible": "Yes", "driver": "Positive"},

        # Negative only
        "Household Monthly Income (Rs)_Rs. 50,000 - Rs. 1 Lakh": {"visible": "Yes", "driver": "Negative"},
        "Household Monthly Income (Rs)_Rs. 50,000 or less": {"visible": "Yes", "driver": "Negative"},
        "Budget_Less than 50 Lakhs": {"visible": "Yes", "driver": "Negative"},
        "Re-engagement count_0_to_2": {"visible": "Yes", "driver": "Negative"},
        "Preferred Property Type_1BHK": {"visible": "Yes", "driver": "Negative"},
        "SFT Range_501 - 1000": {"visible": "Yes", "driver": "Negative"},
        "SFT Range_<500": {"visible": "Yes", "driver": "Negative"},
        "Sum of Unsold inventory_lessthan15.85per": {"visible": "Yes", "driver": "Negative"},
        "DEVELOPERS YOU WORK WITH_STATUS_1_0-9": {"visible": "Yes", "driver": "Negative"},
        "WHICH ARE YOUR STRONG MICRO MARKET_STATUS_1_0to3": {"visible": "Yes", "driver": "Negative"},
        "WHICH OTHER MARKETS DO YOU OPERATE IN_STATUS_1_0-2": {"visible": "Yes", "driver": "Negative"},
        "TARGET SALE FOR THIS FINANCIAL YEAR_STATUS_Out of Range": {"visible": "Yes", "driver": "Negative"},

        # Everything else defaults to BOTH
    }

    # ---------------------------
    # Preprocessing
    # ---------------------------
    df = df.drop(columns=["Quotation_flag"], errors="ignore")
    feature_names = df.columns
    df = df.reindex(columns=feature_names, fill_value=0)

    explainer = joblib.load(r"CP_explainer_jb.joblib")
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ---------------------------
    # Core Logic (UPDATED)
    # ---------------------------
    def classify_shap_row(shap_row, data_row):

        filtered_pairs = [
            (feature, shap_val)
            for feature, shap_val in zip(feature_names, shap_row)
            if data_row[feature] == 1
        ]

        if len(filtered_pairs) == 0:
            return ("", "", "")

        pairs_sorted = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)

        n = len(pairs_sorted)
        split_1 = n // 3
        split_2 = 2 * n // 3

        positive, medium, negative = [], [], []

        for i, (feature, value) in enumerate(pairs_sorted):

            readable = feature_map.get(feature, feature)

            # Apply control
            control = feature_control.get(feature, {"visible": "Yes", "driver": "Both"})

            if control["visible"] != "Yes" or control["driver"] == "":
                continue

            allowed_driver = control["driver"]

            # Bucket assignment
            if i < split_1:
                bucket = "Positive"
            elif i < split_2:
                bucket = "Medium"
            else:
                bucket = "Negative"

            # Driver filter
            if allowed_driver != "Both" and allowed_driver != bucket:
                continue

            # Append
            if bucket == "Positive":
                positive.append(readable)
            elif bucket == "Medium":
                medium.append(readable)
            else:
                negative.append(readable)

        return (
            ", ".join(positive),
            ", ".join(medium),
            ", ".join(negative)
        )

    results = [
        classify_shap_row(shap_values[i], df.iloc[i])
        for i in range(len(df))
    ]

    return pd.DataFrame(
        results,
        columns=["positive_drivers", "medium_drivers", "negative_drivers"]
    )