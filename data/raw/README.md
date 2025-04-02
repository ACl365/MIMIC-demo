# Raw Data

This directory contains the raw MIMIC-III and MIMIC-IV datasets.

## MIMIC-III

The MIMIC-III (Medical Information Mart for Intensive Care III) database is a large, freely-available database comprising de-identified health-related data associated with over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Centre between 2001 and 2012.

### Files

The MIMIC-III dataset includes the following files:

- `ADMISSIONS.csv`: Hospital admission information
- `CALLOUT.csv`: Information about when patients were ready for discharge
- `CAREGIVERS.csv`: Information about caregivers
- `CHARTEVENTS.csv`: Charted events such as vital signs, laboratory tests, etc.
- `CPTEVENTS.csv`: Current Procedural Terminology (CPT) codes
- `D_CPT.csv`: Dictionary of CPT codes
- `D_ICD_DIAGNOSES.csv`: Dictionary of ICD-9 diagnosis codes
- `D_ICD_PROCEDURES.csv`: Dictionary of ICD-9 procedure codes
- `D_ITEMS.csv`: Dictionary of items in CHARTEVENTS
- `D_LABITEMS.csv`: Dictionary of laboratory items
- `DATETIMEEVENTS.csv`: Date and time events
- `DIAGNOSES_ICD.csv`: ICD-9 diagnoses
- `DRGCODES.csv`: Diagnosis Related Group (DRG) codes
- `ICUSTAYS.csv`: ICU stay information
- `INPUTEVENTS_CV.csv`: Input events from CareVue
- `INPUTEVENTS_MV.csv`: Input events from MetaVision
- `LABEVENTS.csv`: Laboratory test results
- `MICROBIOLOGYEVENTS.csv`: Microbiology test results
- `NOTEEVENTS.csv`: Clinical notes
- `OUTPUTEVENTS.csv`: Output events
- `PATIENTS.csv`: Patient information
- `PRESCRIPTIONS.csv`: Medication prescriptions
- `PROCEDUREEVENTS_MV.csv`: Procedure events from MetaVision
- `PROCEDURES_ICD.csv`: ICD-9 procedures
- `SERVICES.csv`: Hospital services
- `TRANSFERS.csv`: Patient transfers

## MIMIC-IV

The MIMIC-IV database is an update to MIMIC-III, containing data from 2008-2019. It includes similar tables to MIMIC-III but with some structural changes and additional data.

### Files

The MIMIC-IV dataset includes the following files:

- `admissions.csv`: Hospital admission information
- `caregiver.csv`: Information about caregivers
- `chartevents.csv`: Charted events such as vital signs, laboratory tests, etc.
- `d_hcpcs.csv`: Dictionary of HCPCS codes
- `d_icd_diagnoses.csv`: Dictionary of ICD diagnosis codes
- `d_icd_procedures.csv`: Dictionary of ICD procedure codes
- `d_labitems.csv`: Dictionary of laboratory items
- `datetimeevents.csv`: Date and time events
- `diagnoses_icd.csv`: ICD diagnoses
- `drgcodes.csv`: Diagnosis Related Group (DRG) codes
- `emar.csv`: Electronic Medication Administration Record
- `hcpcsevents.csv`: HCPCS events
- `icustays.csv`: ICU stay information
- `ingredientevents.csv`: Ingredient events
- `inputevents.csv`: Input events
- `labevents.csv`: Laboratory test results
- `microbiologyevents.csv`: Microbiology test results
- `omr.csv`: Order-Medication-Route
- `outputevents.csv`: Output events
- `patients.csv`: Patient information
- `pharmacy.csv`: Pharmacy information
- `poe.csv`: Provider Order Entry
- `poe_detail.csv`: Provider Order Entry details
- `prescriptions.csv`: Medication prescriptions
- `procedureevents.csv`: Procedure events
- `procedures_icd.csv`: ICD procedures
- `services.csv`: Hospital services
- `transfers.csv`: Patient transfers

## Data Access

The MIMIC datasets are available through PhysioNet. Access requires:

1. Completion of a training course in human subjects research
2. Acceptance of a data use agreement
3. Registration on the PhysioNet website

For more information, visit: https://physionet.org/content/mimiciii/1.4/ and https://physionet.org/content/mimiciv/1.0/

## Citation

If you use the MIMIC-III dataset, please cite:

Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

If you use the MIMIC-IV dataset, please cite:

Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2021). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 8, 1-7.
