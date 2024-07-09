import typing as tp
from math import ceil

import tracdap.rt.api as trac

from wholesale_ifrs9.models.utils.pd_curve_utils import convert_to_dot_notation, _calculate_macro_adjustment, _calculate_PiT_PDs, calculate_weighted_average_curves
from wholesale_ifrs9.models.utils.TransitionMatrix import TransitionMatrix
import wholesale_ifrs9.schemas as schemas

"""
A model that generates monthly forecasts of PD term structures for the three UK portfolios.
"""


class Main(trac.TracModel):

    # Set the model parameters
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:
        return trac.declare_parameters(
            trac.P("reporting_date", trac.DATE, label="Reporting date"),
            trac.P("n_years", trac.INTEGER, label="Number of years of forecast")
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        rating_master_scale_schema = trac.load_schema(schemas, "non_retail_rating_master_scale_schema.csv")
        generic_transition_matrix_schema = trac.load_schema(schemas, "transition_matrix_schema.csv")
        distribution_definition_schema = trac.load_schema(schemas, "distribution_definition_schema.csv")
        gcc_monthly_scenarios_schema = trac.load_schema(schemas, "gcc_monthly_scenarios_schema.csv")
        gcc_macroeconomic_model_coefficients_schema = trac.load_schema(schemas, "gcc_macroeconomic_model_coefficients_schema.csv")

        return {
            "corporate_rating_master_scale": trac.ModelInputSchema(rating_master_scale_schema),
            "corporate_transition_matrix": trac.ModelInputSchema(generic_transition_matrix_schema),
            "gcc_monthly_scenarios": trac.ModelInputSchema(gcc_monthly_scenarios_schema),
            "gcc_distribution_definition": trac.ModelInputSchema(distribution_definition_schema),
            "gcc_macroeconomic_model_coefficients": trac.ModelInputSchema(gcc_macroeconomic_model_coefficients_schema),
        }

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        gcc_corporate_pd_term_structures_schema = trac.load_schema(schemas, "gcc_corporate_pd_term_structures_schema.csv")

        return {
            "gcc_corporate_pd_term_structures": trac.define_output_table(gcc_corporate_pd_term_structures_schema.table.fields, label="GCC corporate PD term structures"),
        }

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")
        n_years = ctx.get_parameter("n_years")

        # The definitions of each grade
        corporate_rating_master_scale = ctx.get_pandas_table("corporate_rating_master_scale")

        # The transition matrix
        # TODO Clean missing values
        corporate_transition_matrix = ctx.get_pandas_table("corporate_transition_matrix")
        gcc_distribution_definition = ctx.get_pandas_table("gcc_distribution_definition")
        gcc_monthly_scenarios = ctx.get_pandas_table("gcc_monthly_scenarios")
        gcc_macroeconomic_model_coefficients = ctx.get_pandas_table("gcc_macroeconomic_model_coefficients")

        total_months = n_years * 12

        median = gcc_distribution_definition["MEDIAN"][0]

        scenarios = {
            "scenarios": {"monthly_scenarios": gcc_monthly_scenarios},
            "totalMonths": total_months,
            "reporting_date": reporting_date,
            "scenario_Names": ['Base', 'Good', 'Better', 'Bad', 'Worse'],
            "macroModel": {"variablesNames": ['KWT Current Account % GDP', 'KWT Inflation Rate L1']}
        }

        scenarios = convert_to_dot_notation(scenarios)

        gcc_monthly_scenarios.rename(columns={
            "KWT_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP": "KWT Current Account % GDP",
            "KWT_INFLATION_RATE_L1": "KWT Inflation Rate L1"
        }, inplace=True)

        gcc_macroeconomic_model_coefficients.rename(columns={
            "KWT_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP": "KWT Current Account % GDP",
            "KWT_INFLATION_RATE_L1": "KWT Inflation Rate L1"
        }, inplace=True)

        lookup = {
            "KWT Current Account % GDP": "KWT_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP",
            "KWT Inflation Rate L1": "KWT_INFLATION_RATE_L1"
        }

        macroImpacts = _calculate_macro_adjustment(
            monthly_scenarios=gcc_monthly_scenarios,
            coefficients=gcc_macroeconomic_model_coefficients,
            variable_names=scenarios.macroModel["variablesNames"],
            scenario_names=scenarios.scenario_Names,
            median=median,
            lookup=lookup
        )

        #############################################################################################
        #                               GCC Corporate portfolio calculation                          #
        #############################################################################################

        logger.info(f"Generating the Corporate PD curves for a {reporting_date} reporting date")

        tm = TransitionMatrix(matrix=corporate_transition_matrix, reporting_date=str(reporting_date), years=ceil(total_months / 12))
        __external_ratings = tm.ratings.tolist().remove("Default")

        corporate_pd_term_structures = _calculate_PiT_PDs(
            tm=tm.interpolated_term_structures,
            mrs=corporate_rating_master_scale,
            external_ratings=tm.ratings.tolist(),
            internal_ratings=corporate_rating_master_scale['Rating'].tolist(),
            scenario_Names=scenarios.scenario_Names,
            macroImpacts=macroImpacts
        )

        corporate_pd_term_structures = calculate_weighted_average_curves(
            pd_term_structures=corporate_pd_term_structures,
            masterRatingScale=corporate_rating_master_scale,
            total_months=total_months
        )

        # Output the datasets
        ctx.put_pandas_table("gcc_corporate_pd_term_structures", corporate_pd_term_structures)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "wholesale_ifrs9/config/generate_pd_curves_for_gcc_corporate_portfolio.yaml", "wholesale_ifrs9/config/sys_config.yaml")
