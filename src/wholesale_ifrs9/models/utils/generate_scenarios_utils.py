from logging import Logger
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy
from dateutil.relativedelta import relativedelta


def calculate_scenario_confidence_interval(row):
    if row["Type"] == "Upside":
        val = 1 - 1 / row["OneInX"]
    elif row["Type"] == "Downside":
        val = 1 / row["OneInX"]
    else:
        val = pd.NA
    return val


def _fix_array_type(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.array(data=df[column].to_list(), dtype=float)

    return df


def _add_dummy_scenarios(all_scenarios: pd.DataFrame, regional_scenarios, logger: Logger, region: str) -> pd.DataFrame:
    # For each unique scenario type and likelihood, check that there is a row for that combination in each regions list of scenarios
    dummy_i = 1
    for index, x in all_scenarios.iterrows():
        if regional_scenarios[(regional_scenarios['OneInX'] == x['OneInX']) & (regional_scenarios['Type'] == x['Type'])].shape[0] == 0:
            regional_scenarios = regional_scenarios.append({'Name': 'Dummy' + str(dummy_i), 'OneInX': x['OneInx'], 'Type': x['Type'], 'USE': False}, ignore_index=True)

    logger.info(f"A total of {dummy_i - 1} dummy scenarios were added to the {region} scenario definitions")

    return regional_scenarios


def _calculate_forecast_default_rates(
        macroeconomic_scenario: pd.DataFrame,
        model_coefficients: pd.DataFrame,
        intercept: float
) -> Tuple[List[float], List[float]]:
    selected_macro_economic_data = macroeconomic_scenario[model_coefficients.columns].to_numpy()

    forecast = selected_macro_economic_data @ model_coefficients.to_numpy().transpose() + intercept

    forecast = forecast.transpose().tolist()[0]

    # This is a time series of the uplift from thr T0 cdf value to the forecast cdf
    linear_uplifts = scipy.stats.norm.cdf(forecast) / scipy.stats.norm.cdf(forecast[0]).tolist()

    return forecast, linear_uplifts


def fix_array_type(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.array(data=df[column].to_list(), dtype=float)

    return df


def add_dummy_scenarios(all_scenarios: pd.DataFrame, regional_scenarios, logger: Logger, region: str) -> pd.DataFrame:
    # For each unique scenario type and likelihood, check that there is a row for that combination in each regions list of scenarios
    dummy_i = 1
    for index, x in all_scenarios.iterrows():
        if regional_scenarios[(regional_scenarios['OneInX'] == x['OneInX']) & (regional_scenarios['Type'] == x['Type'])].shape[0] == 0:
            regional_scenarios = regional_scenarios.append({'Name': 'Dummy' + str(dummy_i), 'OneInX': x['OneInx'], 'Type': x['Type'], 'USE': False}, ignore_index=True)

    logger.info(f"A total of {dummy_i - 1} dummy scenarios were added to the {region} scenario definitions")

    return regional_scenarios


def calculate_forecast_default_rates(
        macroeconomic_scenario: pd.DataFrame,
        model_coefficients: pd.DataFrame,
) -> pd.DataFrame:

    intercept = model_coefficients["Intercept"][0]
    model_coefficients = model_coefficients.copy().drop(columns=["Intercept"])

    # Get just the economic series from the scenario that have model coefficients
    selected_macro_economic_data = macroeconomic_scenario[model_coefficients.columns].to_numpy()

    # Multiply the coefficients by the time series and sum for each time period, add in intercept
    forecast = selected_macro_economic_data @ model_coefficients.to_numpy().transpose() + intercept

    # Add the forecast to the macroeconomic scenario, convert to float, so we can apply the cdf
    # function to it
    macroeconomic_scenario["FORECAST_DEFAULT_RATE"] = list(forecast.flatten())
    macroeconomic_scenario["FORECAST_DEFAULT_RATE"] = macroeconomic_scenario["FORECAST_DEFAULT_RATE"].astype(float)

    # This is a time series of the uplift from thr T0 cdf value to the forecast cdf
    macroeconomic_scenario["SCENARIO_UPLIFT"] = (
            scipy.stats.norm.cdf(macroeconomic_scenario["FORECAST_DEFAULT_RATE"]) / scipy.stats.norm.cdf(macroeconomic_scenario["FORECAST_DEFAULT_RATE"][0])
    )

    return macroeconomic_scenario


def calculate_scenario_confidence_interval(df: pd.DataFrame) -> pd.DataFrame:
    """
    A function that converts a scenario likelihood e.g '1 in 20' into a
    confidence interval that can be used to get the equivalent points on
    a distribution.

    Args:
        df: The scenario definitions with the likelihood values.

    Returns:

        The scenario definitions with the confidence internal added.
    """

    # The bool type is so that the conditions is bool rather than# boolean 
    # type see https://stackoverflow.com/questions/68618078/invalid-entry-0-in-condlist-should-be-boolean-ndarray-using-np-select
    conditions = [
        (df["Type"].str.upper() == "UPSIDE").astype(bool),
        (df["Type"].str.upper() == "DOWNSIDE").astype(bool)
    ]

    choices = [
        1 - 1 / df["OneInX"],
        1 / df["OneInX"]
    ]

    df["Confidence"] = np.select(conditions, choices, default=np.nan)

    return df


def calculate_scenario_weights_and_default_rates(
        scenario_definitions: pd.DataFrame,
        macroeconomic_scenario: pd.DataFrame,
        distribution_fitting_function,
        mu: float,
        sigma: float
) -> pd.DataFrame:
    df = macroeconomic_scenario.copy()

    # Remove the T0 data
    df = df.loc[df["Quarter"] != "Present"]

    # Create a time period from the index of the dataFrame, this assumes that
    # the index is a simple integer row index, do this after the filter so Q1 has index 1
    df = df.rename_axis('Timestep').reset_index(drop=False)

    # Add in the base scenario values, we don't need to calculate anything for the base, so
    # it is a straight copy of the calculated default rates
    df['Scenario_Name'] = "Base"
    df['Weight'] = 0
    df['CI'] = df['FORECAST_DEFAULT_RATE']

    # How likely is each point in the default rate forecast
    df["BASE_CASE_PROBABILITY"] = df['FORECAST_DEFAULT_RATE'].apply(lambda x: distribution_fitting_function.cdf(x, mu, sigma))

    df_temp = df.copy()

    # Now for each scenario we need to calculate the weight that
    # scenario has in the mix and also what the default rate would be
    # for the given likelihood, using the fitted distribution parameters
    for index, scenario_definition in scenario_definitions.iterrows():

        df_temp['Scenario_Name'] = scenario_definition['Name']

        if scenario_definition['Confidence'] <= 0.5:
            # Get worse: right hand tail
            df_temp["SCENARIO_PROBABILITY"] = (1 - scenario_definition['Confidence']) * (1 - df_temp["BASE_CASE_PROBABILITY"]) + df_temp["BASE_CASE_PROBABILITY"]
            df_temp["CI"] = df_temp['SCENARIO_PROBABILITY'].apply(lambda x: distribution_fitting_function.ppf(x, mu, sigma))
            df_temp['Weight'] = 1 - df_temp["CI"].apply(lambda x: distribution_fitting_function.cdf(x, mu, sigma))

        else:
            # Get better: left hand tail
            df_temp["SCENARIO_PROBABILITY"] = -(scenario_definition['Confidence'] * df_temp["BASE_CASE_PROBABILITY"] - df_temp["BASE_CASE_PROBABILITY"])
            df_temp["CI"] = df_temp['SCENARIO_PROBABILITY'].apply(lambda x: distribution_fitting_function.ppf(x, mu, sigma))
            df_temp['Weight'] = df_temp["CI"].apply(lambda x: distribution_fitting_function.cdf(x, mu, sigma))

        df = pd.concat([df, df_temp])

    df.drop(columns=["BASE_CASE_PROBABILITY", "SCENARIO_PROBABILITY"], inplace=True)

    return df


def max_le(vals, val):
    return max([v for v in vals if v < val])


def min_ge(vals, val):
    return min([v for v in vals if v > val])


def calculate_base_scenario_weight(
        df: pd.DataFrame,
        first_upside_scenario: str,
        first_downside_scenario: str
) -> pd.DataFrame:
    keep_list = ["Weight"]
    first_upside_scenario_weights = df.loc[(df["Scenario_Name"] == first_upside_scenario), keep_list]
    # There is a bug in the original code I think where the last row in the forecast is used to
    # calculate the base case weight
    first_upside_scenario_weights = first_upside_scenario_weights.tail(1)
    first_upside_scenario_weights.rename(columns={"Weight": "FIRST_UPSIDE_WEIGHT"}, inplace=True)

    first_downside_scenario_weights = df.loc[(df["Scenario_Name"] == first_downside_scenario), keep_list]
    # There is a bug in the original code I think where the last row in the forecast is used to
    # calculate the base case weight
    first_downside_scenario_weights = first_downside_scenario_weights.tail(1)
    first_downside_scenario_weights.rename(columns={"Weight": "FIRST_DOWNSIDE_WEIGHT"}, inplace=True)

    base_scenario_weights = df.loc[df["Scenario_Name"] == "Base", ["Scenario_Name", "Timestep"]]

    base_scenario_weights = pd.merge(
        base_scenario_weights,
        first_upside_scenario_weights,
        how="cross",
    )

    base_scenario_weights = pd.merge(
        base_scenario_weights,
        first_downside_scenario_weights,
        how='cross',
    )

    base_scenario_weights["BASE_SCENARIO_WEIGHT"] = 1 - base_scenario_weights["FIRST_UPSIDE_WEIGHT"] - base_scenario_weights["FIRST_DOWNSIDE_WEIGHT"]
    base_scenario_weights.drop(columns=["FIRST_UPSIDE_WEIGHT", "FIRST_DOWNSIDE_WEIGHT"], inplace=True)

    df = pd.merge(
        df,
        base_scenario_weights,
        how='left',
        on=["Timestep", "Scenario_Name"],
        validate="one_to_one"
    )

    df.loc[df["Scenario_Name"] == "Base", "Weight"] = df["BASE_SCENARIO_WEIGHT"]
    df.drop(columns=["BASE_SCENARIO_WEIGHT"], inplace=True)

    return df


def normalise_scenario_weights(
        df: pd.DataFrame
) -> pd.DataFrame:
    df["SUM_OF_WEIGHTS"] = df.groupby(["Timestep"])["Weight"].transform(sum)

    df["Weight"] = df["Weight"] / df["SUM_OF_WEIGHTS"]

    sum_of_normalised_weights = df.groupby(["Timestep"])["Weight"].transform(sum).round(4)

    if (sum_of_normalised_weights > 1).sum() > 0:
        raise Exception("Weight error")

    df.drop(columns=["SUM_OF_WEIGHTS"], inplace=True)

    return df


def create_grid_of_macroeconomics(
        macroeconomic_variables_list: List[str],
        historic_macroeconomic_variables: pd.DataFrame,
        macroeconomic_scenario: pd.DataFrame,
        model_coefficients: pd.DataFrame,
        grid_size: int = 1000
):

    macroeconomic_variables_1 = macroeconomic_variables_list[0]

    # Create grid of possible outcomes with directional consistency relative to the first macroeconomic
    # variable

    # step_sizes is a dataframe with the economic series names as the index and the
    # step sizes in the first column
    step_sizes = (
                         historic_macroeconomic_variables[macroeconomic_variables_list].mean(numeric_only=True) +
                         (historic_macroeconomic_variables[macroeconomic_variables_list].std(numeric_only=True) * 10)
                 ) - (
                         historic_macroeconomic_variables[macroeconomic_variables_list].mean(numeric_only=True) -
                         (historic_macroeconomic_variables.std(numeric_only=True) * 10)
                 )

    step_sizes = step_sizes / grid_size

    # Take the ratio of the model coefficients to the first macroeconomic variable
    sign_consistency = (model_coefficients[macroeconomic_variables_list] / model_coefficients[macroeconomic_variables_1][0]) > 0
    sign_consistency = sign_consistency.transpose() > 0
    # change type to series using iloc[]
    sign_consistency = sign_consistency.iloc[:, 0]

    # Get the Q1 forecast economic values as a series
    base_case = macroeconomic_scenario[macroeconomic_variables_list].iloc[1, :]

    # right_bounds is a series of the upper bounds for the macroeconomic
    # series in the grid
    right_bounds = base_case + step_sizes * grid_size / 2 * (
            sign_consistency.to_numpy(dtype=int) - (sign_consistency == False).to_numpy(dtype=int)
    )

    # left_bounds is a series of the lower bounds for the macroeconomic
    # series in the grid
    left_bounds = base_case + step_sizes * grid_size / 2 * (
            (sign_consistency == False).to_numpy(dtype=int) - sign_consistency.to_numpy(dtype=int)
    )

    # Get all the macroeconomic values for all cells in the grid
    macro_economic_value_grid = np.linspace(left_bounds, right_bounds, grid_size * 2 - 1).round(10)
    macro_economic_value_grid = pd.DataFrame(macro_economic_value_grid, columns=macroeconomic_variables_list)

    # Compute predicted CI per grid unit
    macro_economic_value_grid = calculate_forecast_default_rates(
        macroeconomic_scenario=macro_economic_value_grid,
        model_coefficients=model_coefficients,
    )

    macro_economic_value_grid.rename(columns={"FORECAST_DEFAULT_RATE": "Prediction"}, inplace=True)
    macro_economic_value_grid.drop(columns=["SCENARIO_UPLIFT"], inplace=True)

    return macro_economic_value_grid


def find_matching_macroeconomics_for_scenario_default_rates(
    macroeconomic_variables_list: List[str],
    macro_economic_value_grid: pd.DataFrame,
    all_scenarios: pd.DataFrame,
    reporting_date: str
):

    grid_predictions = macro_economic_value_grid["Prediction"].to_numpy().astype(float)
    grid_predictions = grid_predictions.reshape(grid_predictions.size, 1)

    specified_ci = all_scenarios["CI"].to_numpy()
    specified_ci = specified_ci.reshape(1, specified_ci.size)

    diffs = abs(grid_predictions - specified_ci)
    diffs = diffs.argsort(axis=0)

    scenarios = all_scenarios.copy()
    scenarios["Date"] = (pd.to_datetime(reporting_date) + relativedelta(months=3) * scenarios["Timestep"])
    scenarios.drop(columns=macroeconomic_variables_list, inplace=True)

    # The copy is needed to avoid SettingWithCopyWarning warnings
    closest_predictions = macro_economic_value_grid.iloc[diffs[0]].copy()
    closest_predictions.index = range(specified_ci.size)
    closest_predictions["Timestep"] = scenarios["Timestep"]
    closest_predictions["Date"] = scenarios["Date"]
    closest_predictions["Scenario_Name"] = scenarios["Scenario_Name"]

    df = pd.merge(
        closest_predictions,
        scenarios,
        on=["Timestep", "Date", "Scenario_Name"],
        how="left",
        validate="one_to_one"
    )

    # df["Date_str"] = df["Date"].dt.strftime("%d/%m/%Y")
    df["Date_str"] = ""
    return df


def convert_quarterly_scenario_to_monthly(
        macroeconomic_variables_list: List[str],
        total_forecast_months: int,
        reporting_date: str,
        base_case_scenario: pd.DataFrame,
        quarterly_forecasts: pd.DataFrame
):
    """
    Transform quarterly data to monthly using linear interpolation, set self.monthly_scenarios
    Fill gap range(months+1, total_projection_months+1) using the last scenario values
    """

    columns = ["Date", "Scenario", "Weight"]
    columns.extend(macroeconomic_variables_list)

    new_df = pd.DataFrame(columns=columns)
    tmp_df = pd.DataFrame(columns=columns)

    number_of_forecast_quarters = len(base_case_scenario[base_case_scenario["Quarter"] != "Present"]["Quarter"].tolist())

    for scenario_name in quarterly_forecasts["Scenario_Name"].unique().tolist():
        quarterly_scenario_for_scenario = quarterly_forecasts[(quarterly_forecasts['Scenario_Name'] == scenario_name)]
        total_months = 0
        for forecast_quarter in range(1, number_of_forecast_quarters + 1):
            if forecast_quarter == 1:
                loop_start = 0
                current_macroeconomics = base_case_scenario[base_case_scenario["Quarter"] == "Present"]
                previous_macroeconomics = current_macroeconomics[macroeconomic_variables_list]
            else:
                loop_start = 1
                previous_macroeconomics = quarterly_scenario_for_scenario[(quarterly_scenario_for_scenario['Timestep'] == forecast_quarter - 1)][macroeconomic_variables_list]

            previous_macroeconomics.index = [0]
            curr_vals = quarterly_scenario_for_scenario[(quarterly_scenario_for_scenario['Timestep'] == forecast_quarter)][macroeconomic_variables_list]
            curr_vals.index = [0]
            steps = curr_vals.subtract(previous_macroeconomics) / 3

            # interpolation
            for m in range(loop_start, 4):
                record = {"Scenario": scenario_name, "Date": pd.to_datetime(reporting_date) + relativedelta(months=total_months)}
                record = pd.DataFrame(record, index=[0])
                record["Weight"] = quarterly_scenario_for_scenario[(quarterly_scenario_for_scenario['Timestep'] == forecast_quarter)]["Weight"].tolist()[0]
                record[macroeconomic_variables_list] = previous_macroeconomics.add(steps * m)
                # append is deprecated so to avoid warnings we move to concat
                new_df = pd.concat([new_df, record], ignore_index=True)
                total_months += 1

        # add terminal record, pad missing rows in range(months+1, total_forecast_months+1) using resample()
        # number loaded = int(self.projectionMonths)
        l = int(number_of_forecast_quarters * 3)
        if total_forecast_months > int(number_of_forecast_quarters * 3):
            l = int(total_forecast_months)
            record = {"Scenario": scenario_name,
                      "Date": pd.to_datetime(reporting_date) + relativedelta(months=total_forecast_months),
                      "Weight": quarterly_scenario_for_scenario[(quarterly_scenario_for_scenario['Timestep'] == number_of_forecast_quarters)]["Weight"].tolist()[0]}

            record = pd.DataFrame(record, index=[0])
            tmp = quarterly_scenario_for_scenario[(quarterly_scenario_for_scenario['Timestep'] == number_of_forecast_quarters)][macroeconomic_variables_list]
            tmp.index = [0]
            record[macroeconomic_variables_list] = tmp
            # append is deprecated so to avoid warnings we move to concat
            new_df = pd.concat([new_df, record], ignore_index=True)

        new_df.index = new_df["Date"]
        new_df = new_df.resample('1M').ffill()  # .drop_duplicates('Date')
        new_df["Date"] = new_df.index
        new_df.index = range(l + 1)
        # append is deprecated so to avoid warnings we move to concat
        tmp_df = pd.concat([tmp_df, new_df], ignore_index=True)

        new_df = new_df.drop(list(range(0, len(new_df))))
    # tmp_df["Date_str"] = pd.to_datetime(tmp_df['Date']).dt.strftime('%d/%m/%Y')
    tmp_df["Date_str"] = ""
    return tmp_df


def define_first_upside_scenario(df: pd.DataFrame) -> str:

    ds_ci = min_ge(df['Confidence'].tolist(), 0.5)
    return df[df['Confidence'] == ds_ci]['Name'].tolist()[0]


def define_first_downside_scenario(df: pd.DataFrame) -> str:

    us_ci = max_le(df['Confidence'].tolist(), 0.5)
    return df[df['Confidence'] == us_ci]['Name'].tolist()[0]


def extract_macroeconomic_variables(df: pd.DataFrame) -> List[str]:

    macroeconomic_variables_list = df.columns.to_list()

    if "Date" in macroeconomic_variables_list:
        macroeconomic_variables_list.remove("Date")

    if "Intercept" in macroeconomic_variables_list:
        macroeconomic_variables_list.remove("Intercept")

    return macroeconomic_variables_list