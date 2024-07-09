import datetime
import typing as tp
from typing import List, Any, Union

import numpy as np
import pandas as pd
import tracdap.rt.api as trac

import wholesale_ifrs9.schemas as schemas

"""
A model that performs business logic validations on the facility data.
"""


def update_df_and_track_progress(
        df: pd.DataFrame,
        df_temp: pd.DataFrame,
        value_column: Union[str | None],
        test_counter: int,
        id_column: str = "AccountID",
) -> tuple[pd.DataFrame, int]:
    """
    A function that appends a set of validation results to the overall validation summary.

    Args:
        df: The overall validation summary.
        df_temp: The new set of validation results to append.
        value_column: The name of the column to use to get the value that was tested, this is also the
        name added to the 'Field' column.
        test_counter: The test number that the new set of validation results are for.
        id_column: The name of the column to use for the 'ID' column, this should be the unique facility identifier.

    Returns:
        Validation summary with new rows appended.
    """

    if value_column is not None:

        # If checking the ID column we need to cope with when this can't be renamed twice
        if id_column == value_column:
            df_temp.rename(columns={id_column: "ID"}, inplace=True)
            df_temp["Value"] = df_temp["ID"]
        else:
            df_temp.rename(columns={id_column: "ID", value_column: "Value"}, inplace=True)

        df_temp["Field"] = value_column
    
    keep_columns = ["ID", "Field", "Value", "Description"]

    df = pd.concat([df, df_temp[keep_columns]], ignore_index=True)

    test_counter += 1
    return df, test_counter


def find_invalid_values_in_column(
        df: pd.DataFrame,
        column_name: str,
        valid_values: List[Any],
        id_column: str,
) -> pd.DataFrame:
    """
    A function that takes the facility data and searches a column for values not
    in the valid values. Rows with  invalid values are turned into a summary
    dataFrame that can be appended to the overall summary of the tests being
    performed.
    """

    invalid_observations = df[~df[column_name].isin(valid_values)][[column_name, id_column]]

    # Create a dataFrame with summary information on the rows with errors
    invalid_observations.rename(columns={id_column: "ID", column_name: "Value"}, inplace=True)
    invalid_observations["Field"] = column_name

    # !r here returns a string representation of the list
    invalid_observations["Description"] = [
        f"{column_name} (={value}) not in {valid_values!r} as per config file"
        for value in invalid_observations["Value"]
    ]

    return invalid_observations


class Main(trac.TracModel):

    # Set the model parameters
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:
        return trac.declare_parameters(
            trac.P("reporting_date", trac.DATE, label="Reporting date")
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:

        facilities_data_schema = trac.load_schema(schemas, "facilities_data_schema.csv")

        return {"facilities_data": trac.ModelInputSchema(facilities_data_schema)}

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:

        validation_results_schema = [
            trac.F(field_name="ID", field_type=trac.STRING, field_order=0, label="Facility ID", categorical=False),
            trac.F(field_name="Field", field_type=trac.STRING, field_order=1, label="Field", categorical=True),
            trac.F(field_name="Value", field_type=trac.STRING, field_order=2, label="Value", categorical=True),
            trac.F(field_name="Description", field_type=trac.STRING, field_order=3, label="Description", categorical=False),
        ]

        return {"validation_results": trac.define_output_table(validation_results_schema, label="Facility data validation results")}

    def run_model(self, ctx: trac.TracContext):

        # Set up the logger 
        logger = ctx.log()
        
        # Load the input data
        facilities_data = ctx.get_pandas_table("facilities_data")

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")

        logger.info(f"Validation of facilities date is being run for the {reporting_date} reporting date")

        # TODO This is to create some error reports
        facilities_data.loc[facilities_data['AccountID'] == '5', 'AccountID'] = None
        facilities_data.loc[facilities_data['AccountID'] == '8', 'AccountID'] = None
        facilities_data.loc[facilities_data['AccountID'] == '10', 'AccountID'] = None

        # There should be a check that account ID is unique - this is outside the validation in the TNP code
        # if not ds.facilities.records_unique:
        #     raise Exception(f"{ds.account_id} is not Unique")

        valid_values_dict = {
            'regional_pd_parameters': {'UK': ['Corporate', 'SME', 'HNWI'], 'GCC': ['Corporate', 'Retail']},
            'regions': ['UK', 'GCC'],
            'external_ratings': ['Aaa', 'Aa1', 'Aa2', 'Aa3', 'A1', 'A2', 'A3', 'Baa1', 'Baa2', 'Baa3', 'Ba1', 'Ba2',
                                 'Ba3', 'B1', 'B2', 'B3', 'Caa1', 'Caa2', 'Caa3', 'Ca-C'],
            'valid_internal_ratings': {
                'Corporate': ['MRS 1 +', 'MRS 1 flat', 'MRS 1 -', 'MRS 2 +', 'MRS 2 flat', 'MRS 2 -', 'MRS 3 +',
                              'MRS 3 flat', 'MRS 3 -', 'MRS 4 +', 'MRS 4 flat', 'MRS 4 -', 'MRS 5 +', 'MRS 5 flat',
                              'MRS 5 -', 'MRS 6 +', 'MRS 6 flat', 'MRS 6 -', 'MRS 7 +', 'MRS 7 flat', 'MRS 7 -',
                              'MRS 8 +', 'MRS 8 flat', 'MRS 8 -', 'MRS 9 +', 'MRS 9 flat', 'MRS 9 -', 'MRS 10 +',
                              'MRS 10 flat', 'MRS 10 -', 'MRS 11 +', 'MRS 11 flat', 'MRS 11 -', 'MRS 12 +',
                              'MRS 12 flat', 'MRS 12 -', 'MRS 13', 'MRS 14', 'MRS 15', 'MRS 16'],
                'HNWI': ['MRS 1 +', 'MRS 1 flat', 'MRS 1 -', 'MRS 2 +', 'MRS 2 flat', 'MRS 2 -', 'MRS 3 +',
                         'MRS 3 flat', 'MRS 3 -', 'MRS 4 +', 'MRS 4 flat', 'MRS 4 -', 'MRS 5 +', 'MRS 5 flat',
                         'MRS 5 -', 'MRS 6 +', 'MRS 6 flat', 'MRS 6 -', 'MRS 7 +', 'MRS 7 flat', 'MRS 7 -', 'MRS 8 +',
                         'MRS 8 flat', 'MRS 8 -', 'MRS 9 +', 'MRS 9 flat', 'MRS 9 -', 'MRS 10 +', 'MRS 10 flat',
                         'MRS 10 -', 'MRS 11 +', 'MRS 11 flat', 'MRS 11 -', 'MRS 12 +', 'MRS 12 flat', 'MRS 12 -',
                         'MRS 13', 'MRS 14', 'MRS 15', 'MRS 16'],
                'SME': ['MRS 1 +', 'MRS 1 flat', 'MRS 1 -', 'MRS 2 +', 'MRS 2 flat', 'MRS 2 -', 'MRS 3 +', 'MRS 3 flat',
                        'MRS 3 -', 'MRS 4 +', 'MRS 4 flat', 'MRS 4 -', 'MRS 5 +', 'MRS 5 flat', 'MRS 5 -', 'MRS 6 +',
                        'MRS 6 flat', 'MRS 6 -', 'MRS 7 +', 'MRS 7 flat', 'MRS 7 -', 'MRS 8 +', 'MRS 8 flat', 'MRS 8 -',
                        'MRS 9 +', 'MRS 9 flat', 'MRS 9 -', 'MRS 10 +', 'MRS 10 flat', 'MRS 10 -', 'MRS 11 +',
                        'MRS 11 flat', 'MRS 11 -', 'MRS 12 +', 'MRS 12 flat', 'MRS 12 -', 'MRS 13', 'MRS 14', 'MRS 15',
                        'MRS 16'], 'Retail': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
            },
            'segments': ['Commercial Real Estate', 'HNWI', 'Mid Corporate', 'SME', 'Large Corporate', 'Mortgages',
                         'Loans', 'Credit Cards'], 'products': ['Loan', 'Credit Card', 'Mortgage'],
            'property_types': ['House', 'Flat, Block', 'Student Accommodation',
                               'Hotel, Serviced Flat, Mixed, Invested in Real Estate', 'Office'],
            'strengths': ['Invincible', 'Strong', 'Good', 'Satisfactory', 'Poor', 'Very poor'],
            'diffs': ['Sov', 'Non-Sov-Bond', 'Non-Sov-Loan'], 'ipres': ['Non-IPRE Haircut', 'IPRE Haircut'],
            'reporting_date': datetime.datetime(2021, 11, 30, 0, 0), 'reporting_date_formatted': '30/11/2021',
            'payment_types': ['Amortising', 'Revolving'], 'rate_types': ['Floating', 'Fixed'],
            'collateral_types': ['Property', 'Collateral', 'None'], 'other_stage2_flags': [0, 1],
            'other_stage3_flags': [0, 1]
        }

        # This is a dataFrame that we will use to store error messages resulting from the tests
        df = pd.DataFrame(columns=["ID", "Field", "Value", "Description"])

        reporting_date = valid_values_dict["reporting_date"]
        reporting_date_formatted = valid_values_dict["reporting_date_formatted"]

        # These are some lists of valid values for various columns, they are used in multiple places
        # so we destructure them into their own variables.
        valid_regions = valid_values_dict["regions"]
        valid_external_ratings = valid_values_dict["external_ratings"]
        valid_internal_ratings = valid_values_dict["valid_internal_ratings"]
        valid_pd_models = valid_values_dict["regional_pd_parameters"]

        # We use this as an index in logging the tests
        test_counter = 1

        logger.info(f"Running test {test_counter} - checking for missing account IDs")

        df_temp = facilities_data.loc[facilities_data["AccountID"].isnull(), ["AccountID"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = "Identifier (AccountID) is missing"

        logger.info(f"A total of {df_temp.shape[0]} rows had missing account IDs")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "AccountID", test_counter)

        # Now we are going to iterate through a list of columns and check if they contain only their valid values
        for column_name, valid_values in [
            ("Region", valid_regions),
            ("LGDRegion", valid_regions),
            ("Segment", valid_values_dict["segments"]),
            ("Product", valid_values_dict["products"]),
            ("PaymentType", valid_values_dict["payment_types"]),
            ("RateType", valid_values_dict["rate_types"]),
            ("CollateralType", valid_values_dict["collateral_types"]),
        ]:
            logger.info(f"Running test {test_counter} - checking column '{column_name}' for invalid values")

            # Get summary information on any rows that don't have valid values
            df_temp = find_invalid_values_in_column(
                df=facilities_data,
                column_name=column_name,
                valid_values=valid_values,
                id_column="AccountID"
            )

            logger.info(f"A total of {df_temp.shape[0]} rows had invalid '{column_name}' values")

            # Append the test results to the overall summary
            df, test_counter = update_df_and_track_progress(df, df_temp, None, test_counter)

        logger.info(f"Running test {test_counter} - identifying PD models that can't be checked due to invalid regions")

        # Find rows with invalid regions
        valid_regions_condition = facilities_data["Region"].isin(valid_regions)

        # Add a variable with the result of the check - True is bad
        facilities_data["check0"] = np.where(valid_regions_condition, False, True)

        df_temp = facilities_data.loc[~valid_regions_condition, ["AccountID", "PDModel", "Region"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"PDModel (={pd_model}) validity cannot be assessed. This is a direct consequence of the following error: 'Region (={region}) not in {valid_regions!r} as per config file'" for pd_model, region in zip(df_temp["PDModel"], df_temp["Region"])
        ]

        logger.info(f"A total of {df_temp.shape[0]} PD models could not be validated due to invalid regions")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "PDModel", test_counter)

        logger.info(f"Running test {test_counter} - checking PD models of valid regions")

        # Get a list of the unique regions in the facility data
        regions_in_facility_data = facilities_data["Region"].unique()
        # Invalid regions are any in the facility data that are not in the model definitions
        invalid_regions = list(set(regions_in_facility_data).difference(valid_regions))
        # Create a key for each invalid region, this is just making sure that the key exists
        for region in invalid_regions:
            valid_pd_models[region] = []

        # Merge on a list of valid PD models by region, we added missing regions to this list
        # which will map to an empty list
        facilities_data["valid_pd_models"] = facilities_data["Region"].map(valid_pd_models)

        # TODO Sadly this can't be vectorised as it is, if you join the list on, then convert to a
        #  string joined by |, then check if that the PdModel is contained in that list, then that
        #  would be a fast version
        facilities_data["in_valid_pd_models"] = facilities_data.apply(
            lambda row: (row["PDModel"] in row["valid_pd_models"]),
            axis=1,
        )

        # Create boolean mask for invalid PDModel check
        invalid_pd_models_condition = (facilities_data["valid_pd_models"].isnull()) | ((~facilities_data["in_valid_pd_models"]) & valid_regions_condition)

        # Add a variable with the result of the check - True is bad
        facilities_data["check1"] = np.where(invalid_pd_models_condition, True, False)

        # Find rows with invalid PD models, but that have valid regions
        df_temp = facilities_data.loc[invalid_pd_models_condition, ["AccountID", "PDModel", "Region"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"PDModel (={pd_model}) should be in {valid_pd_models[region]} given region (={region}) as per config file"
            for region, pd_model in zip(df_temp["Region"], df_temp["PDModel"])
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid PD models (but valid regions)")
        
        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "PDModel", test_counter)

        for column_name in ["InternalRatingOrigination", "InternalRatingPresent"]:
            
            logger.info(f"Running test {test_counter} - identifying '{column_name}' values that can't be checked due to invalid PD models")
            
            # Find rows with invalid PD models
            df_temp = facilities_data.loc[facilities_data["check1"], ["AccountID", column_name, "Region", "PDModel"]]

            # Add a message about what was found wrong with the data
            df_temp["Description"] = [
                f"{column_name} (={column_value}) validity cannot be assessed. This is a direct consequence of the following error: 'PDModel (={pd_model}) should be in {valid_pd_models[region]} given region (={region}) as per config file'"
                for region, pd_model, column_value in zip(
                    df_temp["Region"], df_temp["PDModel"], df_temp[column_name]
                )
            ]

            logger.info(f"A total of {df_temp.shape[0]} '{column_name}' values could not be validated due to invalid PD models")
            
            # Append the test results to the overall summary
            df, test_counter = update_df_and_track_progress(df, df_temp, column_name, test_counter)

        logger.info(f"Running test {test_counter} - checking invalid payment frequencies")

        # Create boolean mask for invalid PaymentType/PaymentFrequency check
        payment_frequency_condition = (facilities_data["PaymentType"] == "Amortising") & (~facilities_data["PaymentFrequency"].isin([1, 3, 6, 12]))

        # Add a variable with the result of the check - True is bad
        facilities_data["check2"] = np.where(payment_frequency_condition, True, False)

        # Find rows with invalid PaymentType/PaymentFrequency models
        df_temp = facilities_data.loc[facilities_data["check2"], ["AccountID", "PaymentFrequency"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"PaymentFrequency (={freq}) not in [1, 3, 6, 12] as per model requirement for amortising products"
            for freq in df_temp["PaymentFrequency"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid payment frequencies (amortising facilities only))")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "PaymentFrequency", test_counter)

        logger.info(f"Running test {test_counter} - checking for missing open dates")

        # TODO TRAC can provide these as dates
        facilities_data["o_date"] = pd.to_datetime(facilities_data["OpenDate"], errors="coerce", dayfirst=True)

        # Find rows with missing open dates
        df_temp = facilities_data.loc[facilities_data["o_date"].isnull(), ["AccountID", "OpenDate"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"OpenDate (={open_date}) is not a valid date"
            for open_date in df_temp["OpenDate"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had missing open dates")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "OpenDate", test_counter)

        logger.info(f"Running test {test_counter} - checking for missing maturity dates on non-revolving facilities")

        # TODO TRAC can provide these as dates
        facilities_data["m_date"] = pd.to_datetime(facilities_data["MaturityDate"], errors="coerce", dayfirst=True
                                                 )
        missing_maturity_date_condition = (facilities_data["m_date"].isnull()) & (facilities_data["PaymentType"] != "Revolving")

        # Add a variable with the result of the check - True is bad
        facilities_data["check3"] = np.where(missing_maturity_date_condition, True, False)

        # Find rows for revolving facilities with missing maturity dates
        df_temp = facilities_data.loc[missing_maturity_date_condition, ["AccountID", "MaturityDate"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"MaturityDate (={maturity_date}) is not a valid date"
            for maturity_date in df_temp["MaturityDate"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} non-revolving facilities had missing maturity dates")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "MaturityDate", test_counter)

        logger.info(f"Running test {test_counter} - checking for open dates occurring after the reporting date for non-revolving facilities")

        opened_after_reporting_date_condition = (facilities_data["o_date"] > reporting_date) & (~facilities_data["o_date"].isnull()) & (facilities_data["PaymentType"] != "Revolving")

        # Add a variable with the result of the check - True is bad
        facilities_data["check4"] = np.where(opened_after_reporting_date_condition, True, False)

        # Find rows for revolving facilities with open dates after the reporting date
        df_temp = facilities_data.loc[opened_after_reporting_date_condition, ["AccountID", "OpenDate"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"OpenDate {open_date} is greater than reporting date {reporting_date_formatted}"
            for open_date in df_temp["OpenDate"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} non-revolving facilities had open dates after the reporting date")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "OpenDate", test_counter)

        logger.info(f"Running test {test_counter} - checking for maturity dates occurring before the reporting date for non-revolving facilities")

        matured_before_reporting_date = (
                (facilities_data["m_date"] <= reporting_date) &
                (~facilities_data["m_date"].isnull()) &
                (facilities_data["PaymentType"] != "Revolving")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check5"] = np.where(matured_before_reporting_date, True, False)

        # Find rows for revolving facilities with maturity dates before the reporting date
        df_temp = facilities_data.loc[matured_before_reporting_date, ["AccountID", "MaturityDate"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"MaturityDate {maturity_date} is less than reporting date {reporting_date_formatted}"
            for maturity_date in df_temp["MaturityDate"]
        ]

        logger.info(
            f"A total of {df_temp.shape[0]} non-revolving facilities had maturity dates before the reporting date")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "MaturityDate", test_counter)

        logger.info(
            f"Running test {test_counter} - checking for maturity dates occurring before open dates for non-revolving facilities")

        maturity_date_before_open_date_condition = (
                (facilities_data["m_date"] <= facilities_data["o_date"]) &
                (~facilities_data["o_date"].isnull()) &
                (facilities_data["PaymentType"] != "Revolving")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check6"] = np.where(maturity_date_before_open_date_condition, True, False)

        # Find rows for revolving facilities with maturity dates before the open date
        df_temp = facilities_data.loc[maturity_date_before_open_date_condition, ["AccountID", "MaturityDate", "o_date"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"MaturityDate {maturity_date} is less than open date {open_date.strftime('%Y/%m/%d')}"
            for open_date, maturity_date in zip(
                df_temp["o_date"], df_temp["MaturityDate"]
            )
        ]

        logger.info(
            f"A total of {df_temp.shape[0]} non-revolving facilities had maturity dates before open dates")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "MaturityDate", test_counter)

        logger.info(
            f"Running test {test_counter} - checking for missing or negative limits for revolving facilities")

        # TODO conversions not needed in TRAC
        # facility_data["limit"] = pd.to_numeric(facility_data["Limit"], errors="coerce")

        invalid_limits_condition = (
                (facilities_data["PaymentType"] == "Revolving") &
                ((facilities_data["Limit"].isna()) | (facilities_data["Limit"] <= 0))
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check7"] = np.where(invalid_limits_condition, True, False)

        # Find rows with invalid limits
        df_temp = facilities_data.loc[invalid_limits_condition, ["AccountID", "Limit"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"Limit (={limit}) should be positive number for 'Revolving' payment types"
            for limit in df_temp["Limit"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} revolving facilities had missing or negative limits")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "Limit", test_counter)

        logger.info(
            f"Running test {test_counter} - checking for missing or negative limits for revolving facilities")

        invalid_exposure_condition = (facilities_data["Exposure"].isna()) | (facilities_data["Exposure"] <= 0)

        # Add a variable with the result of the check - True is bad
        facilities_data["check8"] = np.where(invalid_exposure_condition, True, False)

        # Find rows that have invalid exposures
        df_temp = facilities_data.loc[invalid_exposure_condition, ["AccountID", "Exposure"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"Exposure (={exposure}) should be positive number - check input has no formatting if it is"
            for exposure in df_temp["Exposure"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} revolving facilities had missing or negative exposures")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "Exposure", test_counter)

        logger.info(f"Running test {test_counter} - identifying bullet payment value that can't be checked due to invalid exposures")

        # Find rows that have invalid exposures (again)
        df_temp = facilities_data.loc[invalid_exposure_condition, ["AccountID", "Exposure", "BulletPayment"]]

        # Add some additional summary information about bullet payments not being able to be validated
        # due to invalid exposures. Since we used the df_temp from above we have to re
        df_temp["Description"] = [
            f"BulletPayment (={bullet_payment}) validity cannot be assessed. This is a direct consequence of the following error: 'Exposure (={exposure}) should be positive number - check input has no formatting if it is'"
            for bullet_payment, exposure in zip(
                df_temp["BulletPayment"], df_temp["Exposure"]
            )
        ]

        logger.info(f"A total of {df_temp.shape[0]} bullet payments values could not be validated due to invalid exposures")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "BulletPayment", test_counter)

        logger.info(f"Running test {test_counter} - checking for missing or negative RWA")

        invalid_rwa_condition = (facilities_data["RWA"].isna()) | (facilities_data["RWA"] <= 0)

        # Add a variable with the result of the check - True is bad
        facilities_data["check9"] = np.where(invalid_rwa_condition, True, False)

        # Find rows for with invalid RWAs
        df_temp = facilities_data.loc[invalid_rwa_condition, ["AccountID", "RWA"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"RWA (={rwa}) should be positive number - check input has no formatting if it is"
            for rwa in df_temp["RWA"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} revolving facilities had missing or negative RWAs")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "RWA", test_counter)

        logger.info(
            f"Running test {test_counter} - checking for missing, negative limits or very high interest rates")

        # Lower bound condition is <0 as some products can be interest free
        invalid_interest_rates_condition = (
                (facilities_data["InterestRate"].isna()) |
                (facilities_data["InterestRate"] < 0) |
                (facilities_data["InterestRate"] >= 1)
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check10"] = np.where(invalid_interest_rates_condition, True, False)

        # Find rows with invalid interest rates
        df_temp = facilities_data.loc[invalid_interest_rates_condition, ["AccountID", "InterestRate"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"InterestRate (={interest_rate}) should be positive number between 0 and 1 - check input has no formatting if it is"
            for interest_rate in df_temp["InterestRate"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid interest rates")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "InterestRate", test_counter)

        logger.info(
            f"Running test {test_counter} - checking for invalid external rating values (at origination)")

        invalid_origination_external_ratings_condition = (
                (~facilities_data["ExternalRatingOrigination"].isnull()) &
                (~facilities_data["ExternalRatingPresent"].isnull()) &
                (~facilities_data["ExternalRatingOrigination"].isin(valid_external_ratings))
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check11"] = np.where(invalid_origination_external_ratings_condition, True, False)

        # Find rows with invalid external ratings at origination
        df_temp = facilities_data.loc[invalid_origination_external_ratings_condition, ["AccountID", "ExternalRatingOrigination"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"ExternalRatingOrigination (={external_rating}) is not in external ratings list configured in config file"
            for external_rating in df_temp["ExternalRatingOrigination"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid external ratings (at origination)")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "ExternalRatingOrigination", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid external rating values (present)")

        invalid_present_external_ratings_condition = (
                (~facilities_data["ExternalRatingOrigination"].isnull()) &
                (~facilities_data["ExternalRatingPresent"].isnull()) &
                (~facilities_data["ExternalRatingPresent"].isin(valid_external_ratings))
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check12"] = np.where(invalid_present_external_ratings_condition, True, False)

        # Find rows with invalid present external ratings
        df_temp = facilities_data.loc[
            invalid_origination_external_ratings_condition, ["AccountID", "ExternalRatingPresent"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"ExternalRatingPresent (={external_rating}) is not in external ratings list configured in config file"
            for external_rating in df_temp["ExternalRatingPresent"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid external ratings (present)")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "ExternalRatingPresent", test_counter)

        logger.info(f"Running test {test_counter} - checking for PD models without any rating values (at origination)")

        facilities_data["valid_internal_ratings"] = "['']"

        not_missing_region_and_pd_mask = ~(facilities_data["Region"].isnull() | facilities_data["PDModel"].isnull())

        internal_rating_dictionary = {
            key: valid_internal_ratings.get(key)
            for key in facilities_data.loc[not_missing_region_and_pd_mask, "PDModel"].unique()
            if key in valid_internal_ratings
        }

        mask = not_missing_region_and_pd_mask & facilities_data["PDModel"].isin(internal_rating_dictionary.keys())

        facilities_data.loc[mask, "valid_internal_ratings"] = facilities_data.loc[mask, "PDModel"].map(internal_rating_dictionary)

        # Find rows with invalid internal ratings at origination
        df_temp = facilities_data.loc[facilities_data["valid_internal_ratings"].isnull(), ["AccountID", "InternalRatingOrigination", "Region", "PDModel"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"InternalRatingOrigination cannot be checked due to invalid region, PDModel pair ({region}, {pd_model})"
            for region, pd_model in zip(df_temp["Region"], df_temp["PDModel"])
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows PD models without valid any values (at origination)")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "InternalRatingOrigination", test_counter)

        logger.info(f"Running test {test_counter} - checking for PD models without any rating values (present)")

        # Find rows with invalid present internal ratings
        df_temp = facilities_data.loc[
            facilities_data["valid_internal_ratings"].isnull(), ["AccountID", "InternalRatingPresent", "Region",
                                                               "PDModel"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"InternalRatingPresent cannot be checked due to invalid region, PDModel pair ({region}, {pd_model})"
            for region, pd_model in zip(df_temp["Region"], df_temp["PDModel"])
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had PD models without any rating values (present)")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "InternalRatingPresent", test_counter)

        for column_name in ["InternalRatingOrigination", "InternalRatingPresent"]:

            logger.info(f"Running test {test_counter} - checking for PD models without valid rating values (present)")

            # Does the row have a list of ratings associated with the PD model and is the value for the internal
            # rating in the list
            facilities_data["is_in_valid_internal_ratings"] = facilities_data.apply(
                lambda row: (
                        (row[column_name] in row["valid_internal_ratings"])
                        and (row["PDModel"] in row["valid_pd_models"])
                ),
                axis=1,
            )

            # Add a message about what was found wrong with the data
            valid_internal_rating_condition = (
                    (~facilities_data["valid_internal_ratings"].isnull()) &
                    (~facilities_data["is_in_valid_internal_ratings"]) &
                    # check1 is for invalid PD models
                    (~facilities_data["check1"])
            )

            # Add a variable with the result of the check - True is bad
            facilities_data[f"check_{column_name}"] = np.where(valid_internal_rating_condition, True, False)

            # Find rows with invalid present internal ratings
            df_temp = facilities_data.loc[valid_internal_rating_condition, ["AccountID", column_name, "Region", "PDModel"]]

            # Add a message about what was found wrong with the data
            df_temp["Description"] = [
                f"{column_name} (={internal_rating}) not valid for Region (={region}) and PDmodel (={pd_model}) configuration"
                for internal_rating, region, pd_model in zip(
                    df_temp[column_name],
                    df_temp["Region"],
                    df_temp["PDModel"],
                )
            ]

            logger.info(f"A total of {df_temp.shape[0]} rows had invalid '{column_name}' values for their PD model and/or region")

            # Append the test results to the overall summary
            df, test_counter = update_df_and_track_progress(df, df_temp, column_name, test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid bullet payments for amortising facilities")

        # TODO this is not needed in TRAC
        # facility_data["BulletPayment"] = np.where(
        #     facility_data["BulletPayment"].isnull(),
        #     "0",
        #     facility_data["BulletPayment"],
        # )
        #
        # facility_data["bullet"] = pd.to_numeric(
        #     facility_data["BulletPayment"], errors="coerce"
        # )

        facilities_data["BulletPayment"].fillna(0, inplace=True)

        # Is exposure is not missing and the facility is amortising do we have a non-missing, non-negative
        # bullet payment that is less than the exposure
        invalid_bullet_payment_condition = (
                (~facilities_data["Exposure"].isnull()) &
                (
                    (facilities_data["BulletPayment"].isnull()) |
                    (facilities_data["BulletPayment"] < 0) |
                    (facilities_data["BulletPayment"] > facilities_data["Exposure"])
                ) &
                (facilities_data["PaymentType"] == "Amortising")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check14"] = np.where(invalid_bullet_payment_condition, True, False)

        # Find rows with invalid bullet payments
        df_temp = facilities_data.loc[invalid_bullet_payment_condition, ["AccountID", "BulletPayment", "Exposure"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"BulletPayment (={bullet_payment}) should be a non-negative number if payment type is 'Amortising' and be less than or equal to the exposure (={exposure}); check input has no formatting if it is"
            for bullet_payment, exposure in zip(
                df_temp["BulletPayment"], df_temp["Exposure"]
            )
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid bullet payments")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "BulletPayment", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid days past due")

        invalid_days_past_due_condition = (
                (facilities_data["DaysPastDue"].isnull()) |
                (facilities_data["DaysPastDue"] < 0) |
                (facilities_data["DaysPastDue"].round() != facilities_data["DaysPastDue"])
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check15"] = np.where(invalid_days_past_due_condition, True, False)

        # Find rows with invalid days past due
        df_temp = facilities_data.loc[invalid_days_past_due_condition, ["AccountID", "DaysPastDue"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"DaysPastDue (={dpd}) should be a non-negative integer check; input has no formatting if it is"
            for dpd in df_temp["DaysPastDue"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid days past due values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "DaysPastDue", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid valuation values")

        invalid_valuation_condition = (((facilities_data["Valuation"].isnull()) | (facilities_data["Valuation"] < 0)) &
                                      (facilities_data["CollateralType"] != "None"))

        # Add a variable with the result of the check - True is bad
        facilities_data["check16"] = np.where(invalid_valuation_condition, True, False)

        # Find rows with invalid valuations
        df_temp = facilities_data.loc[invalid_valuation_condition, ["AccountID", "Valuation"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"Valuation (={valuation}) should be a non-negative number if collateral type not equal to 'None' - check input has no formatting if it is"
            for valuation in df_temp["Valuation"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid valuations")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "Valuation", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid IPRE values")

        invalid_ipre_condition = (
                (~facilities_data["IPRE"].isin(valid_values_dict["ipres"])) &
                (facilities_data["CollateralType"] == "Property")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check17"] = np.where(invalid_ipre_condition, True, False)

        # Find rows with invalid IPRE values
        df_temp = facilities_data.loc[invalid_ipre_condition, ["AccountID", "IPRE"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"IPRE (={ipre}) should be as per config file if collateral type equal to 'Property'"
            for ipre in df_temp["IPRE"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid IPRE values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "IPRE", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid property types")

        invalid_property_types_condition = (
                (~facilities_data["PropertyType"].isin(valid_values_dict["property_types"])) &
                (facilities_data["CollateralType"] == "Property")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check18"] = np.where(invalid_property_types_condition, True, False)

        # Find rows with invalid property types
        df_temp = facilities_data.loc[invalid_property_types_condition, ["AccountID", "PropertyType"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"PropertyType (={property_type}) should be as per config file if collateral type equal to 'Property'"
            for property_type in df_temp["PropertyType"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid property types")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "PropertyType", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid index valuations")

        invalid_index_valuations_condition = (
                ((facilities_data["IndexValuation"].isnull()) | (facilities_data["IndexValuation"] <= 0)) &
                (facilities_data["CollateralType"] == "Property")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check19"] = np.where(invalid_index_valuations_condition, True, False)

        # Find rows with invalid index valuations
        df_temp = facilities_data.loc[invalid_index_valuations_condition, ["AccountID", "IndexValuation"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"IndexValuation (={index_valuation}) should be positive number if type equal to 'Property' - check input has no formatting if it is"
            for index_valuation in df_temp["IndexValuation"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid index valuations")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "IndexValuation", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid instrument differentiator values")

        invalid_differentiator_condition = (
                (~facilities_data["InstrumentDifferentiator"].isin(valid_values_dict["diffs"])) &
                (facilities_data["CollateralType"] == "None")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check20"] = np.where(invalid_differentiator_condition, True, False)

        # Find rows with invalid instrument differentiator values
        df_temp = facilities_data.loc[invalid_differentiator_condition, ["AccountID", "InstrumentDifferentiator"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"InstrumentDifferentiator (={diff}) should be as per config file if collateral type equal to 'None'"
            for diff in df_temp["InstrumentDifferentiator"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid instrument differentiator values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "InstrumentDifferentiator", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid collateral strength values")

        invalid_collateral_strength_condition = (
                (~facilities_data["CollateralStrength"].isin(valid_values_dict["strengths"])) &
                (facilities_data["CollateralType"] == "Collateral")
        )

        # Add a variable with the result of the check - True is bad
        facilities_data["check21"] = np.where(invalid_collateral_strength_condition, True, False)

        # Find rows with invalid instrument differentiator values
        df_temp = facilities_data.loc[invalid_collateral_strength_condition, ["AccountID", "CollateralStrength"]]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"CollateralStrength (={strength}) should be as per config file if collateral type equal to 'Collateral'"
            for strength in df_temp["CollateralStrength"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid collateral strength values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "CollateralStrength", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid (other) stage 2 flags")

        # Find rows with invalid stage 2 flags values
        df_temp = facilities_data.loc[
            ~facilities_data["OtherStage2Flag"].isin(valid_values_dict["other_stage2_flags"]), ["AccountID", "OtherStage2Flag"]
        ]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"OtherStage2Flag (={other_stage2_flag}) should be equal to 0 or 1"
            for other_stage2_flag in df_temp["OtherStage2Flag"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid (other) stage 2 flag values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "OtherStage2Flag", test_counter)

        logger.info(f"Running test {test_counter} - checking for invalid (other) stage 3 flags")

        # Find rows with invalid stage 2 flags values
        df_temp = facilities_data.loc[
            ~facilities_data["OtherStage3Flag"].isin(valid_values_dict["other_stage3_flags"]), ["AccountID", "OtherStage3Flag"]
        ]

        # Add a message about what was found wrong with the data
        df_temp["Description"] = [
            f"OtherStage3Flag (={other_stage3_flag}) should be equal to 0 or 1"
            for other_stage3_flag in df_temp["OtherStage3Flag"]
        ]

        logger.info(f"A total of {df_temp.shape[0]} rows had invalid (other) stage 3 flag values")

        # Append the test results to the overall summary
        df, test_counter = update_df_and_track_progress(df, df_temp, "OtherStage3Flag", test_counter)

        # Do some cleaning
        df["Description"] = df["Description"].replace({"'None'": "null", "nan": "null"}, regex=False)
        df["Value"] = df["Value"].astype(str).replace("nan", "null")

        # Bring all the records for each ID together
        df.sort_values("ID", inplace=True)

        # Output the dataset
        ctx.put_pandas_table("validation_results", df)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "wholesale_ifrs9/config/validate_facility_data.yaml", "wholesale_ifrs9/config/sys_config.yaml")
