import typing as tp

import pandas as pd
from scipy.stats import norm
from scipy import stats
import tracdap.rt.api as trac

from wholesale_ifrs9.models.utils.generate_scenarios_utils import (
    fix_array_type, calculate_forecast_default_rates, calculate_base_scenario_weight,
    calculate_scenario_weights_and_default_rates,
    create_grid_of_macroeconomics,
    find_matching_macroeconomics_for_scenario_default_rates,
    convert_quarterly_scenario_to_monthly,
    calculate_scenario_confidence_interval,
    normalise_scenario_weights,
    define_first_upside_scenario,
    define_first_downside_scenario,
    extract_macroeconomic_variables
)
import wholesale_ifrs9.schemas as schemas

"""
A model that generates quarterly and monthly forecast scenarios and information about the distribution used to
generate them.
"""


class Main(trac.TracModel):

    # Set the model parameters
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:
        return trac.declare_parameters(
            trac.P("reporting_date", trac.DATE, label="Reporting date"),
            trac.P("n_years", trac.INTEGER, label="Number of years"),
            trac.P("distribution_fitting_method", trac.STRING, label="Distribution fitting method", default_value="gumbel_r")
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        scenario_definitions_schema = trac.load_schema(schemas, "scenario_definitions_schema.csv")
        uk_scenarios_schema = trac.load_schema(schemas, "uk_scenario_schema.csv")
        historic_default_rates_schema = trac.load_schema(schemas, "historic_default_rates_schema.csv")
        uk_macroeconomic_model_coefficients_schema = trac.load_schema(schemas, "uk_macroeconomic_model_coefficients_schema.csv")
        uk_historic_macroeconomic_variables_schema = trac.load_schema(schemas, "uk_historic_macroeconomic_variables_schema.csv")

        return {
            "uk_scenario_definitions": trac.ModelInputSchema(scenario_definitions_schema),
            "uk_base_case_scenario": trac.ModelInputSchema(uk_scenarios_schema),
            "uk_historic_default_rates": trac.ModelInputSchema(historic_default_rates_schema),
            "uk_macroeconomic_model_coefficients": trac.ModelInputSchema(uk_macroeconomic_model_coefficients_schema),
            "uk_historic_macroeconomic_variables": trac.ModelInputSchema(uk_historic_macroeconomic_variables_schema)
        }

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        uk_quarterly_scenarios_schema = trac.load_schema(schemas, "uk_quarterly_scenarios_schema.csv")
        uk_monthly_scenarios_schema = trac.load_schema(schemas, "uk_monthly_scenarios_schema.csv")
        distribution_definition_schema = trac.load_schema(schemas, "distribution_definition_schema.csv")

        return {
            "uk_quarterly_scenarios": trac.define_output_table(uk_quarterly_scenarios_schema.table.fields, label="UK quarterly scenario"),
            "uk_monthly_scenarios": trac.define_output_table(uk_monthly_scenarios_schema.table.fields, label="UK monthly scenario"),
            "uk_distribution_definition": trac.define_output_table(distribution_definition_schema.table.fields, label="UK monthly scenario")
        }

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")
        n_years = ctx.get_parameter("n_years")
        distribution_fitting_method = ctx.get_parameter("distribution_fitting_method")

        logger.info(f"Generating the UK model scenarios for a {reporting_date} reporting date")

        # Load the input data
        # The names of the scenarios, their type and their likelihood
        uk_scenario_definitions = ctx.get_pandas_table("uk_scenario_definitions")

        # Historic macroeconomic time series
        uk_historic_macroeconomic_variables = ctx.get_pandas_table("uk_historic_macroeconomic_variables")

        # Historic default rates
        uk_historic_default_rates = ctx.get_pandas_table("uk_historic_default_rates")

        # Macroeconomic model coefficients
        uk_macroeconomic_model_coefficients = ctx.get_pandas_table("uk_macroeconomic_model_coefficients")

        # The macroeconomic forecasts for the base scenario
        uk_base_case_scenario = ctx.get_pandas_table("uk_base_case_scenario")

        # TODO this is a conversion from the more modern float array type to the older pandas float array type
        uk_historic_macroeconomic_variables = fix_array_type(uk_historic_macroeconomic_variables, ["UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP", "UK_UNEMPLOYMENT"])
        uk_base_case_scenario = fix_array_type(uk_base_case_scenario, ["UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP", "UK_UNEMPLOYMENT"])

        # The scenario definitions the name type and likelihood of each scenario
        uk_scenario_definitions['USE'] = True

        # Add on the confidence interval associated with the scenario likelihood
        uk_scenario_definitions = calculate_scenario_confidence_interval(df=uk_scenario_definitions)

        # The method norm.ppf() takes a percentage and returns a standard deviation
        # multiplier for what value that percentage occurs at
        default_rate_distribution_multipliers = [norm.ppf(default_rate) for default_rate in uk_historic_default_rates['DEFAULT_RATE']]

        # Using the getattr function allows us to get the fitting function by name
        distribution_fitting_function = getattr(stats, distribution_fitting_method)

        # Fit the historic default rate distribution
        (mu, sigma) = distribution_fitting_function.fit(default_rate_distribution_multipliers)
        median = distribution_fitting_function.mean(mu, sigma)

        coefficients_without_time_series = list(set(uk_macroeconomic_model_coefficients.columns) - set(uk_base_case_scenario.columns) - {"Intercept"})

        if len(coefficients_without_time_series) > 0:
            raise ValueError(f"The macroeconomic scenario is missing time series for the following coefficients : {coefficients_without_time_series}")

        uk_base_case_scenario = calculate_forecast_default_rates(
            macroeconomic_scenario=uk_base_case_scenario,
            model_coefficients=uk_macroeconomic_model_coefficients,
        )

        uk_all_scenarios = calculate_scenario_weights_and_default_rates(
            scenario_definitions=uk_scenario_definitions,
            macroeconomic_scenario=uk_base_case_scenario,
            distribution_fitting_function=distribution_fitting_function,
            mu=mu,
            sigma=sigma,
        )

        # Use the confidence intervals to find the first upside and downside scenarios
        # These are used to set the weight of the base scenario
        first_downside_scenario = define_first_downside_scenario(df=uk_scenario_definitions)
        first_upside_scenario = define_first_upside_scenario(df=uk_scenario_definitions)

        # Calculate the base scenario weight as this is set to 0 when we calculate the default rates
        uk_all_scenarios = calculate_base_scenario_weight(
            df=uk_all_scenarios,
            first_downside_scenario=first_downside_scenario,
            first_upside_scenario=first_upside_scenario,
        )

        # Normalise across all scenario so the weights sum to 1
        uk_all_scenarios = normalise_scenario_weights(df=uk_all_scenarios)

        # Define which variables are being used in the econometric models as independent variables
        macroeconomic_variables_list = extract_macroeconomic_variables(df=uk_historic_macroeconomic_variables)

        # Look at the historic distribution of each macroeconomic variable and
        # create a grid of values covering the full range and with small step sizes
        # For each combination calculate the corresponding default rate according to the
        # econometric model
        grid_of_macroeconomic_values_and_default_rates = create_grid_of_macroeconomics(
            macroeconomic_variables_list=macroeconomic_variables_list,
            historic_macroeconomic_variables=uk_historic_macroeconomic_variables,
            macroeconomic_scenario=uk_base_case_scenario,
            model_coefficients=uk_macroeconomic_model_coefficients,
        )

        # Add the corresponding macroeconomics that are consistent with the
        # default rates for each scenario
        uk_quarterly_scenarios = find_matching_macroeconomics_for_scenario_default_rates(
            macroeconomic_variables_list=macroeconomic_variables_list,
            macro_economic_value_grid=grid_of_macroeconomic_values_and_default_rates,
            reporting_date=reporting_date,
            all_scenarios=uk_all_scenarios,
        )

        # Interpolate the quarterly forecasts to monthly
        uk_monthly_scenarios = convert_quarterly_scenario_to_monthly(
            macroeconomic_variables_list=macroeconomic_variables_list,
            total_forecast_months=n_years * 12,
            reporting_date=reporting_date,
            base_case_scenario=uk_base_case_scenario,
            quarterly_forecasts=uk_quarterly_scenarios
        )

        # Put all the information about the distributions into a dataFrame
        distribution_details = pd.DataFrame({
            "MU": [mu],
            "SIGMA": [sigma],
            "DISTRIBUTION_LOW_VALUE": [-3.5],
            "DISTRIBUTION_HIGH_VALUE": [0],
            "DISTRIBUTION_RANGE": [71],
            "DISTRIBUTION_INTERVAL": [0.05],
            "MEDIAN": [median]
        })

        uk_monthly_scenarios["Date"] = uk_monthly_scenarios["Date"].astype(str)
        uk_quarterly_scenarios["Date"] = uk_quarterly_scenarios["Date"].astype(str)

        # Output the datasets
        ctx.put_pandas_table("uk_quarterly_scenarios", uk_quarterly_scenarios)
        ctx.put_pandas_table("uk_monthly_scenarios", uk_monthly_scenarios)
        ctx.put_pandas_table("uk_distribution_definition", distribution_details)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "wholesale_ifrs9/config/generate_uk_scenarios.yaml", "wholesale_ifrs9/config/sys_config.yaml")
