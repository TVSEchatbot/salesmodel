import pandas as pd
import numpy as np
import joblib
import gc
import os
import io
import math
import re
import shap
from sklearn.preprocessing import StandardScaler
pd.set_option('future.no_silent_downcasting', True)

# Flattening the data

#flatenning json
def Transformer2(data):

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
    #my_list['Re_engagement_count']= my_list.filter(regex=("reEnquiries_(.*)_Id")).shape[1]

    # leadid = my_list['leadDetail_Enquiry_Ref_No_']
    # Enquiryrefno = my_list['leadDetail_Lead_ID']

    return my_list

# Digital model function - Event 1

def digital_pipeline2(df):


    #Rename columns to standardized names
    column_mapping = {
    "leadDetail_Re-engagement_count":"Re_engagement_count",
    "leadDetail_Medium":"Medium",
    "leadDetail_Market_/_Channel":"MarketChannel",
    "AIMLObject_Total_Incoming_Answered_Calls__c":"Connected_Incoming_Bucket",
    "AIMLObject_Total_Outgoing_Answered_Calls__c":"Connected_Outgoing_Bucket",
    "leadDetail_SFT_Range":"Sqft_Range_New",
    "leadDetail_Interested_in":"Apartment_Type",
    "leadDetail_Applicant_-_Nature_of_Occupation":"Applicant_Occupation_Bucket",
    "leadDetail_Preferred_Property_Type":"Unit_Type_new",
    "leadDetail_Appointment_Initiated_Date":"Svform_date",
    "leadDetail_Refer_a_family_or_friend":"Refer Your Relative",
    "leadDetail_Budget":"Budget",
    "leadDetail_Reason_for_Purchase":"Purchase Reason",
    "leadDetail_Is_this_your_First_Home_Purchase?":"First Home",
    "leadDetail_Household_Monthly_Income_(Rs)":"Income_Group",
    "leadDetail_Current_Accomodation_Type":"Accomodation",
    "leadDetail_Applicant_-_Industry":"Applicant_Industry",
    "leadDetail_How_Many_Kids?":"Kids Strength",
    "leadDetail_Marital_Status":"Marital Status",
    "leadDetail_Age":"age_prospect",
    "leadDetail_Gender":"Gender",
    "leadDetail_Enquiry_Ref_No_":"Enquiry ref no",
    "leadDetail_Created_Date":"Lead_Created_Date",
    "leadDetail_Lead_ID":"Lead_ID",
    "leadDetail_Category":"Residential_Status_Bucket",
    "leadDetail_Call_remarks":"Call_remarks"
    #"quotation_Quotation_count": "Quotation_count"
    }
    df.rename(columns=column_mapping, inplace=True)

    #remove the square brackets
    df['MarketChannel']=df['MarketChannel'].str.replace(r"[\[\]]", "", regex=True).str.strip()
    df['Medium']=df['Medium'].str.replace(r"[\[\]]", "", regex=True).str.strip()
    
    #Filter only relevant channels
    if 'MarketChannel' in df.columns:
        df['MarketChannel'] = df['MarketChannel'].astype(str).str.strip().str.title()
        df = df[df['MarketChannel'].isin([
            'Cti', 'Digital', 'Domestic-Btl', 'Domestic-Digital', 
            'Domestic-Direct & Atl', 'Google', 'Lead Nurturing', 
            'Nri-Btl', 'Nri-Direct & Atl', 'Website'
        ])]

        
    # campaign bucketing
    subsourceconditions = [
    df['MarketChannel'].str.contains('Domestic-Direct & ATL|NRI-Direct & ATL', na=False, case=False),
    df['MarketChannel'].str.contains('Domestic-BTL|NRI-BTL', na=False, case=False),
    df['Medium'].str.contains('Lead Nurturing|CTI', na=False, case=False),
    df['Medium'].str.contains('Affiliate|Aggregators', na=False, case=False),
    df['Medium'].str.contains('Facebook|fb', na=False, case=False),
    df['Medium'].str.contains('Google', na=False, case=False),
    df['Medium'].str.contains('Linkedin', na=False, case=False),
    df['Medium'].str.contains('Native Ads|native', na=False, case=False),
    df['Medium'].str.contains('Publisher', na=False, case=False),
    df['Medium'].str.contains('SMS & Email', na=False, case=False),
    df['Medium'].str.contains('Website|webchat|chatbot|bot', na=False, case=False),
    df['Medium'].str.contains('Others', na=False, case=False)
    ]
    subsourcevalues = [
    'Direct & ATL', 'BTL', 'Lead nurturing', 'Aggregators', 'Facebook',
    'Google', 'Linkedin', 'Native Ads', 'Publishers', 'SMS & Email',
    'Website / Webchat', 'Digital_others'
    ]
    df['presalescampaign_Bucket'] = np.select(subsourceconditions, subsourcevalues, default='Digital_others')

    #Age grouping
    # Create age group using nested np.where
    df['age_group'] = np.where(df['age_prospect'] < 30, 'Less than 30',
                       np.where((df['age_prospect'] >= 30) & (df['age_prospect'] < 40), '30 to 40',
                       np.where((df['age_prospect'] >= 40) & (df['age_prospect'] < 50), '40 to 50',
                       np.where(df['age_prospect'] > 60, 'More than 60', '40 to 50'))))

    #applicant industry
    df['applicant_industry_group'] = (
    df['Applicant_Industry']
    .replace(r'^\s*$', np.nan, regex=True)
    .fillna('not_revealed'))


    #current accomodation type
    df['Current_accomodation_group'] =  np.where(df['Accomodation'].isin(["Rented","rent-house"]), "Rent-house",
                                                 np.where(df['Accomodation'].isna(), "Rent-house", df['Accomodation']))             


    
    #BHK mapping
    def map_bhk(value):
        if pd.isna(value):
            return 'Not Revealed'
        last_entry = value.split(';')[-1].strip().lower()
    
        if 'not revealed' in last_entry:
            return 'Not Revealed'
        elif '1' in last_entry:
            return '1 BHK'
        elif '2' in last_entry and '2.5' not in last_entry:
            return '2 BHK'
        elif '2.5' in last_entry:
            return '2 BHK'
        elif '3' in last_entry:
            return '3 BHK'
        elif '4' in last_entry:
            return '4 BHK'
        elif '5' in last_entry:
            return '5 BHK'
        else:
            return 'Others'

    # Apply mapping
    df['BHK_Group'] = df['Unit_Type_new'].apply(map_bhk)   

    #Property type mapping
    def map_property_type(value):
        if pd.isna(value) or value.strip() == '':
            return 'Apartments'
        
        value = value.lower()
        if 'plot' in value:
            return 'Plots'
        elif 'villa' in value or 'row house' in value:
            return 'Villas'
        else:
            return 'Apartments'

    # Apply function
    df['Apartment_Type_Bucket'] = df['Apartment_Type'].apply(map_property_type)

    # Convert column to string (to handle mixed types), then map values
    df['Kids Strength_grp'] = df['Kids Strength'].astype(str).str.strip().map({
        '0': 0, '0.0': 0,
        '1': 1, '1.0': 1,
        '2': 2, '2.0': 2,
        '3': 3, '3.0': 3,
        '3 or more': 3,
        'nan' : 0
    })
    
    # Impute missing or unrecognized values as 0
    df['Kids Strength_grp'] = df['Kids Strength_grp'].fillna(0).astype(int)


    # Accepted bucketization
    #df['age_prospect'] = df['age_prospect'].apply(lambda age: '18-38' if age <= 38 else '39-45' if age <= 45 else '46 and above')
    
    df['Connected_Incoming_Bucket'] = df['Connected_Incoming_Bucket'].apply(lambda x: '0-1' if x <= 1 else '2 or more')
    df['Connected_Outgoing_Bucket'] = df['Connected_Outgoing_Bucket'].apply(lambda x: '0-1' if x <= 1 else 'More than 2')
    df['Kids_Strength'] = pd.to_numeric(df['Kids Strength'], errors='coerce').apply(lambda x: '0-1' if x <= 1 else 'More than 2')
    df['Re_engagement_Bucket'] = df['Re_engagement_count'].apply(lambda x: '0-1' if x <= 1 else '2 or more')

    
    #Imputation for missing values
    df['Connected_Incoming_Bucket'] = df['Connected_Incoming_Bucket'].fillna(0)
    df['Connected_Outgoing_Bucket'] = df['Connected_Outgoing_Bucket'].fillna(0)
    #df['age_prospect'] = df['age_prospect'].fillna(41)

    #Budget grouping
    def map_price_range(value):
        # Handle non-string or missing values
        if pd.isnull(value):
            return "Not Revealed"
        
        value = str(value).lower().replace("?", "-").replace(" to ", "-").replace("—", "-")
        
        # Standard bucket mappings
        if any(kw in value for kw in ["not revealed", "unknown"]):
            return "Not Revealed"
        elif "less than" in value or "<" in value or "below" in value or "under" in value:
            return "<50 Lakhs"
        elif "30" in value and ("<30" in value or "less" in value):
            return "<50 Lakhs"
        elif any(x in value for x in ["30 - 40", "40 - 50", "50 - 60", "60 - 70", "50 lakh", "50 lakhs"]):
            return "50 Lakhs - 75 Lakhs"
        elif any(x in value for x in ["70 - 80", "80 - 90", "75", "90 lakh", "75 l", "75 lakhs"]):
            return "75 Lakhs - 1 Crore"
        elif any(x in value for x in ["1 crore", "1.2", "1.4", "1.5", "1.6", "1.8"]):
            return "1 Crore - 2 Crores"
        elif any(x in value for x in ["2 crore", "2.2", "2.4", "2.6", "2.8"]):
            return "2 Crores - 3 Crores"
        elif any(x in value for x in ["3 crore", "3.2", "3.5"]):
            return "3 Crores - 4 Crores"
        elif any(x in value for x in ["4 crore", "4.5"]):
            return "4 Crores - 5 Crores"
        elif any(x in value for x in ["5 crore", "6", "7", "8", "9"]):
            return "5 Crores - 10 Crores"
        elif any(x in value for x in ["10", "12", "15", "more than"]):
            return "More than 10 Crores"
        else:
            return "Not Revealed"
    
    # Apply mapping
    df['Budget_group'] = df['Budget'].apply(map_price_range)
    

    #Feature engineering
    df['Lead_Created_Date'] = pd.to_datetime(df['Lead_Created_Date'], format="%d/%m/%Y")
    df['Svform_date'] = pd.to_datetime(df['Svform_date'], format="%Y-%m-%d")
    df['Days_AP_Lead'] = (df['Svform_date'] - df['Lead_Created_Date']).dt.days
    df['Days_AP_Lead'] = df['Days_AP_Lead'].apply(lambda x: 0 if pd.notnull(x) and x < 0 else x)
    # median_days = df[df['Days_AP_Lead'] > 0]['Days_AP_Lead'].median()
    # df['Days_AP_Lead'] = df['Days_AP_Lead'].apply(lambda x: median_days if pd.isna(x) or x == 0 else x)
    df['Days_AP_Lead_Bucket'] = df['Days_AP_Lead'].apply(lambda x: '0-2' if x <= 2 else '3-51' if x <= 51 else '52 and above')
    df['Lead_Created_Day'] = df['Lead_Created_Date'].dt.day_name()
    df['Site_Visit_Day'] = df['Svform_date'].dt.day_name()
    df['First Home_group'] = np.where(df['First Home'].isin(['yes','Yes']),'Yes',np.where(df['First Home'].isna(),'Not revealed',df['First Home']))
    #df[['Days_AP_Lead', 'age_prospect', 'Connected_Incoming_Bucket', 'Connected_Outgoing_Bucket']] = df[['Days_AP_Lead', 'age_prospect', 'Connected_Incoming_Bucket', 'Connected_Outgoing_Bucket']].fillna(0).astype(int)
    df['Re_engagement_Bucket'] = df['Re_engagement_count'].apply(lambda x: '0-1' if x <= 1 else '2 or more')
    df['Martial_status_kids'] = df['Marital Status'].astype(str) + " - " + df['Kids Strength_grp'].astype(str)
    df['Purchasereason_Firsthome_Interaction'] = df['First Home_group'].astype(str) + '_' + df['Purchase Reason'].astype(str)
    
    
    #processing the RM feedback
    df['Call_remarks'] = df['Call_remarks'].fillna('').str.lower()

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
        df[group] = df['Call_remarks'].str.contains(pattern, case=False, na=False).astype(int)


    # #Quotation data
    # df['Quotation_flag']=np.where(df['Quotation_count']>0,1,0)
    # df['Quotation_flag']=df['Quotation_flag'].astype(int)

    #Choosing the required columns
    columns_to_choose =['Gender', 'Sqft_Range_New', 'Residential_Status_Bucket', 'presalescampaign_Bucket', 'age_group', 'applicant_industry_group', 'Current_accomodation_group', 'BHK_Group', 'Budget_group', 'Days_AP_Lead_Bucket', 'Lead_Created_Day', 'Site_Visit_Day', 'Re_engagement_Bucket', 'Martial_status_kids', 'Time to think', 'Future visit', 'Virtual meet', 'Purchase intent', 'Project details', 'Payment plans or cost sheet', 'Type of purchase', 'Within_Budget', 'Decision maker', 'Flat preference', 'Facing preference', 'Not interested', 'Seeking property type', 'Budget constraints', 'Vaastu constraints', 'Amenities constraints', 'Property constraints']

    df = df[columns_to_choose]

    #One hot encoding the required columns
    columns_to_encode = ['Gender', 'Sqft_Range_New', 'Residential_Status_Bucket', 'presalescampaign_Bucket', 'age_group', 'applicant_industry_group', 'Current_accomodation_group', 'BHK_Group', 'Re_engagement_Bucket', 'Budget_group', 'Days_AP_Lead_Bucket', 'Lead_Created_Day', 'Site_Visit_Day', 'Martial_status_kids']

    df_encoded = pd.get_dummies(df, columns=columns_to_encode)

    #Check if the required columns are present or else impute and make it as zeroes

    required_cols  =['Time to think','Future visit','Virtual meet','Purchase intent','Project details','Payment plans or cost sheet','Type of purchase','Within_Budget','Decision maker','Flat preference','Facing preference','Not interested','Seeking property type','Budget constraints','Vaastu constraints','Amenities constraints','Property constraints','Gender_Male','Sqft_Range_New_1251 - 1500','Sqft_Range_New_1501 - 1750','Sqft_Range_New_1751 - 2000','Sqft_Range_New_2001 - 2250','Sqft_Range_New_2251 - 2500','Sqft_Range_New_2501 - 2750','Sqft_Range_New_2751 - 3000','Sqft_Range_New_3001 - 3250','Sqft_Range_New_3251 - 3500','Sqft_Range_New_3501 - 3750','Sqft_Range_New_3751 - 4000','Sqft_Range_New_4001 - 4250','Sqft_Range_New_4251 - 4500','Sqft_Range_New_4501 - 4750','Sqft_Range_New_4751 - 5000','Sqft_Range_New_501 - 1000','Sqft_Range_New_<500','Sqft_Range_New_> 5001','Sqft_Range_New_Not Revealed','Residential_Status_Bucket_NRI','presalescampaign_Bucket_BTL','presalescampaign_Bucket_Digital_others','presalescampaign_Bucket_Direct & ATL','presalescampaign_Bucket_Facebook','presalescampaign_Bucket_Google','presalescampaign_Bucket_Lead nurturing','presalescampaign_Bucket_Linkedin','presalescampaign_Bucket_Native Ads','presalescampaign_Bucket_Publishers','presalescampaign_Bucket_Website / Webchat','age_group_40 to 50','age_group_Less than 30','age_group_More than 60','applicant_industry_group_Banking and Financial services','applicant_industry_group_Business','applicant_industry_group_Education/Academics','applicant_industry_group_FMCG','applicant_industry_group_Government','applicant_industry_group_Hospitality','applicant_industry_group_IT/ Software','applicant_industry_group_Manufacturing','applicant_industry_group_Medical','applicant_industry_group_Others','applicant_industry_group_Real Estate','applicant_industry_group_Retail services','applicant_industry_group_Retired','applicant_industry_group_Software','applicant_industry_group_Telecommunication','applicant_industry_group_not_revealed','Current_accomodation_group_company-provided-house','Current_accomodation_group_self-own-house','BHK_Group_2 BHK','BHK_Group_3 BHK','BHK_Group_4 BHK','BHK_Group_5 BHK','BHK_Group_Not Revealed','BHK_Group_Others','Re_engagement_Bucket_2 or more','Budget_group_2 Crores - 3 Crores','Budget_group_3 Crores - 4 Crores','Budget_group_4 Crores - 5 Crores','Budget_group_5 Crores - 10 Crores','Budget_group_50 Lakhs - 75 Lakhs','Budget_group_75 Lakhs - 1 Crore','Budget_group_<50 Lakhs','Budget_group_Not Revealed','Days_AP_Lead_Bucket_3-51','Days_AP_Lead_Bucket_52 and above','Lead_Created_Day_Monday','Lead_Created_Day_Saturday','Lead_Created_Day_Sunday','Lead_Created_Day_Thursday','Lead_Created_Day_Tuesday','Lead_Created_Day_Wednesday','Site_Visit_Day_Monday','Site_Visit_Day_Saturday','Site_Visit_Day_Sunday','Site_Visit_Day_Thursday','Site_Visit_Day_Tuesday','Site_Visit_Day_Wednesday','Martial_status_kids_Married - 1','Martial_status_kids_Married - 2','Martial_status_kids_Married - 3','Martial_status_kids_Married with children - 0','Martial_status_kids_Married with children - 1','Martial_status_kids_Married with children - 2','Martial_status_kids_Married with children - 3','Martial_status_kids_Single - 0','Martial_status_kids_Single - 1','Martial_status_kids_Single - 2','Martial_status_kids_Single - 3','Martial_status_kids_nan - 0']

    # Function to ensure columns exist
    def ensure_columns(dfenc, required_cols):
        # Find missing columns
        missing_cols = [col for col in required_cols if col not in dfenc.columns]
        
        if missing_cols:
            # Create DataFrame of zeros with same index
            zeros = pd.DataFrame(0, index=dfenc.index, columns=missing_cols)
            # Concatenate in one go
            dfenc = pd.concat([dfenc, zeros], axis=1)
        return dfenc

    # Apply
    df_final = ensure_columns(df_encoded, required_cols)

    #reorder the columns

    neededcols= ['Time to think',
    'Future visit',
    'Virtual meet',
    'Purchase intent',
    'Project details',
    'Payment plans or cost sheet',
    'Type of purchase',
    'Within_Budget',
    'Decision maker',
    'Flat preference',
    'Facing preference',
    'Not interested',
    'Seeking property type',
    'Budget constraints',
    'Vaastu constraints',
    'Amenities constraints',
    'Property constraints',
    'Gender_Male',
    'Sqft_Range_New_1251 - 1500',
    'Sqft_Range_New_1501 - 1750',
    'Sqft_Range_New_1751 - 2000',
    'Sqft_Range_New_2001 - 2250',
    'Sqft_Range_New_2251 - 2500',
    'Sqft_Range_New_2501 - 2750',
    'Sqft_Range_New_2751 - 3000',
    'Sqft_Range_New_3001 - 3250',
    'Sqft_Range_New_3251 - 3500',
    'Sqft_Range_New_3501 - 3750',
    'Sqft_Range_New_3751 - 4000',
    'Sqft_Range_New_4001 - 4250',
    'Sqft_Range_New_4251 - 4500',
    'Sqft_Range_New_4501 - 4750',
    'Sqft_Range_New_4751 - 5000',
    'Sqft_Range_New_501 - 1000',
    'Sqft_Range_New_<500',
    'Sqft_Range_New_> 5001',
    'Sqft_Range_New_Not Revealed',
    'Residential_Status_Bucket_NRI',
    'presalescampaign_Bucket_BTL',
    'presalescampaign_Bucket_Digital_others',
    'presalescampaign_Bucket_Direct & ATL',
    'presalescampaign_Bucket_Facebook',
    'presalescampaign_Bucket_Google',
    'presalescampaign_Bucket_Lead nurturing',
    'presalescampaign_Bucket_Linkedin',
    'presalescampaign_Bucket_Native Ads',
    'presalescampaign_Bucket_Publishers',
    'presalescampaign_Bucket_Website / Webchat',
    'age_group_40 to 50',
    'age_group_Less than 30',
    'age_group_More than 60',
    'applicant_industry_group_Banking and Financial services',
    'applicant_industry_group_Business',
    'applicant_industry_group_Education/Academics',
    'applicant_industry_group_FMCG',
    'applicant_industry_group_Government',
    'applicant_industry_group_Hospitality',
    'applicant_industry_group_IT/ Software',
    'applicant_industry_group_Manufacturing',
    'applicant_industry_group_Medical',
    'applicant_industry_group_Others',
    'applicant_industry_group_Real Estate',
    'applicant_industry_group_Retail services',
    'applicant_industry_group_Retired',
    'applicant_industry_group_Software',
    'applicant_industry_group_Telecommunication',
    'applicant_industry_group_not_revealed',
    'Current_accomodation_group_company-provided-house',
    'Current_accomodation_group_self-own-house',
    'BHK_Group_2 BHK',
    'BHK_Group_3 BHK',
    'BHK_Group_4 BHK',
    'BHK_Group_5 BHK',
    'BHK_Group_Not Revealed',
    'BHK_Group_Others',
    'Re_engagement_Bucket_2 or more',
    'Budget_group_2 Crores - 3 Crores',
    'Budget_group_3 Crores - 4 Crores',
    'Budget_group_4 Crores - 5 Crores',
    'Budget_group_5 Crores - 10 Crores',
    'Budget_group_50 Lakhs - 75 Lakhs',
    'Budget_group_75 Lakhs - 1 Crore',
    'Budget_group_<50 Lakhs',
    'Budget_group_Not Revealed',
    'Days_AP_Lead_Bucket_3-51',
    'Days_AP_Lead_Bucket_52 and above',
    'Lead_Created_Day_Monday',
    'Lead_Created_Day_Saturday',
    'Lead_Created_Day_Sunday',
    'Lead_Created_Day_Thursday',
    'Lead_Created_Day_Tuesday',
    'Lead_Created_Day_Wednesday',
    'Site_Visit_Day_Monday',
    'Site_Visit_Day_Saturday',
    'Site_Visit_Day_Sunday',
    'Site_Visit_Day_Thursday',
    'Site_Visit_Day_Tuesday',
    'Site_Visit_Day_Wednesday',
    'Martial_status_kids_Married - 1',
    'Martial_status_kids_Married - 2',
    'Martial_status_kids_Married - 3',
    'Martial_status_kids_Married with children - 0',
    'Martial_status_kids_Married with children - 1',
    'Martial_status_kids_Married with children - 2',
    'Martial_status_kids_Married with children - 3',
    'Martial_status_kids_Single - 0',
    'Martial_status_kids_Single - 1',
    'Martial_status_kids_Single - 2',
    'Martial_status_kids_Single - 3',
    'Martial_status_kids_nan - 0']
    
    df_final = df_final[neededcols]
    
    df_final =df_final[neededcols]
    

    # Digisales_rfc_model = joblib.load(r'LS_Event1.pkl')
    # Digisales_pred = Digisales_rfc_model.predict_proba(df_final)[:, 1]

    return df_final

#df_processed = digitalmodel(my_list)

def digital_model_predict2(df):
    Digisales_rfc_model = joblib.load(r'LS_Event1_sckikitupdate.pkl')
    Digisales_pred = Digisales_rfc_model.predict_proba(df)[:, 1]
    classification = 'High Likely to Buy' if Digisales_pred >= 0.017063 else 'Less likely to Buy'
    return Digisales_pred, classification

def digital_shap_scores(df):

    import joblib
    import pandas as pd

    # ---------------------------
    # 1. Feature Mapping Dictionary
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
    # Preprocessing
    # ---------------------------
    df = df.drop(columns=["Quotation_flag"], errors="ignore")
    feature_names = df.columns
    df = df.reindex(columns=feature_names, fill_value=0)

    explainer = joblib.load(r"Digital_explainer_jb.joblib")
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

            # ✅ Replace with readable name
            readable = feature_map.get(feature, feature)

            if i < split_1:
                positive.append(readable)
            elif i < split_2:
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