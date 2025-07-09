
cd = {}

cd['c0']  = { 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'Facility Name_White Plains Hospital Center', 'Facility Name_Staten Island University Hosp-South', 'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 'Facility Name_University Hospital of Brooklyn', 'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 'Emergency Department Indicator_Y', 'Payment Typology 1_Medicare', 'Race_Black/African American', 'Zip Code_112', 'Age Group_18 to 29', 'APR Risk of Mortality_Moderate', 'APR Medical Surgical Description_Medical' }
cd['c1']  = { 'Hospital County_Bronx', 'Facility Name_UPSTATE University Hospital at Community General', 'Facility Name_University Hospital - Stony Brook Southampton Hospital', 'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 'Facility Name_The Unity Hospital of Rochester', 'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 'Facility Name_Syosset Hospital', 'Hospital County_Manhattan', 'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 'Facility Name_UPMC Chautauqua at WCA', 'Facility Name_Wyoming County Community Hospital', 'Ethnicity_Spanish/Hispanic', 'Race_White', 'Payment Typology 3_Self-Pay', 'Age Group_70 or Older', 'Ethnicity_Not Span/Hispanic', 'APR Risk of Mortality_Minor', 'APR Medical Surgical Description_Not Applicable', 'CCS Procedure Code_0' }
cd['c2']  = { 'Facility Name_St. Marys Healthcare', 'Facility Name_University Hospital SUNY Health Science Center', 'Facility Name_Vassar Brothers Medical Center', 'Facility Name_Mount Sinai Beth Israel', 'Facility Name_United Memorial Medical Center North Street Campus', 'Facility Name_Wyckoff Heights Medical Center', 'Payment Typology 2_Medicaid', 'Gender_F', 'Type of Admission_Elective', 'APR Severity of Illness Code_1', 'APR MDC Code_14', 'APR MDC Code_5', 'APR Risk of Mortality_Major', 'CCS Procedure Code_231' }
cd['c3']  = { 'Facility Name_Mount Sinai Hospital', 'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 'Emergency Department Indicator_N', 'Facility Name_Women And Childrens Hospital Of Buffalo', 'Facility Name_Westchester Medical Center', 'Facility Name_University Hospital', 'Facility Name_Staten Island University Hosp-North', 'Facility Name_Westfield Memorial Hospital Inc', 'Facility Name_Strong Memorial Hospital', 'Facility Name_TLC Health Network Lake Shore Hospital', 'Facility Name_Woodhull Medical & Mental Health Center', 'Facility Name_The Burdett Care Center', 'Payment Typology 1_Private Health Insurance', 'Patient Disposition_Home w/ Home Health Services', 'Race_Other Race', 'Zip Code_100', 'Patient Disposition_Home or Self Care', 'Zip Code_104', 'Payment Typology 1_Medicaid', 'Gender_M', 'Payment Typology 2_Self-Pay', 'Age Group_30 to 49', 'Age Group_0 to 17', 'Age Group_50 to 69', 'Length of Stay', 'Type of Admission_Emergency', 'APR Severity of Illness Code_3', 'APR Severity of Illness Code_2', 'APR Medical Surgical Description_Surgical', 'APR MDC Code_4' }


cd['hospital'] = {
    'Hospital County_Bronx', 
    'Hospital County_Manhattan',
    'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 
    'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 
    'Facility Name_Mount Sinai Beth Israel', 
    'Facility Name_Mount Sinai Hospital', 
    'Facility Name_St. Marys Healthcare', 
    'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 
    'Facility Name_Staten Island University Hosp-North', 
    'Facility Name_Staten Island University Hosp-South', 
    'Facility Name_Strong Memorial Hospital', 
    'Facility Name_Syosset Hospital', 
    'Facility Name_TLC Health Network Lake Shore Hospital', 
    'Facility Name_The Burdett Care Center', 
    'Facility Name_The Unity Hospital of Rochester', 
    'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 
    'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 
    'Facility Name_UPMC Chautauqua at WCA', 
    'Facility Name_UPSTATE University Hospital at Community General', 
    'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 
    'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 
    'Facility Name_United Memorial Medical Center North Street Campus', 
    'Facility Name_University Hospital', 
    'Facility Name_University Hospital - Stony Brook Southampton Hospital', 
    'Facility Name_University Hospital SUNY Health Science Center', 
    'Facility Name_University Hospital of Brooklyn', 
    'Facility Name_Vassar Brothers Medical Center', 
    'Facility Name_Westchester Medical Center', 
    'Facility Name_Westfield Memorial Hospital Inc', 
    'Facility Name_White Plains Hospital Center', 
    'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 
    'Facility Name_Women And Childrens Hospital Of Buffalo', 
    'Facility Name_Woodhull Medical & Mental Health Center', 
    'Facility Name_Wyckoff Heights Medical Center', 
    'Facility Name_Wyoming County Community Hospital', 
    'Emergency Department Indicator_N', 
    'Emergency Department Indicator_Y' 
    #'Permanent Facility Id'
    }

cd['patient'] = {
    'Gender_F', 
    'Gender_M',
    'Payment Typology 3_Self-Pay',
    'Race_Black/African American', 
    'Race_Other Race', 
    'Race_White',
    'Payment Typology 1_Medicaid', 
    'Payment Typology 1_Medicare', 
    'Payment Typology 1_Private Health Insurance', 
    'Zip Code_100', 
    'Zip Code_104', 
    'Zip Code_112', 
    'Payment Typology 2_Medicaid', 
    'Payment Typology 2_Self-Pay', 
    'Ethnicity_Not Span/Hispanic', 
    'Ethnicity_Spanish/Hispanic',  
    'Age Group_0 to 17', 
    'Age Group_18 to 29', 
    'Age Group_30 to 49', 
    'Age Group_50 to 69', 
    'Age Group_70 or Older', 
    'Patient Disposition_Home or Self Care', 
    'Patient Disposition_Home w/ Home Health Services', 
    }

cd['admission'] = {
    'Length of Stay', 
    'Type of Admission_Elective', 
    'Type of Admission_Emergency',
    } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

cd['illness'] = { 
    'APR MDC Code_4', 
    'APR MDC Code_5', 
    'APR MDC Code_14', 
    'APR Risk of Mortality_Major', 
    'APR Risk of Mortality_Minor', 
    'APR Risk of Mortality_Moderate', 
    'CCS Procedure Code_0', 
    'CCS Procedure Code_231', 
    'APR Severity of Illness Code_1', 
    'APR Severity of Illness Code_2', 
    'APR Severity of Illness Code_3', 
    'APR Medical Surgical Description_Medical', 
    'APR Medical Surgical Description_Not Applicable', 
    'APR Medical Surgical Description_Surgical',
    }

for k in cd:
    print(k, len(cd[k]))
