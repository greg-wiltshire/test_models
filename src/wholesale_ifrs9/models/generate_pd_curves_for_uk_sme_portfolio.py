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
            trac.P("n_years", trac.INTEGER, label="Number of years")
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        rating_master_scale_schema = trac.load_schema(schemas, "non_retail_rating_master_scale_schema.csv")
        transition_matrix_schema = trac.load_schema(schemas, "transition_matrix_schema.csv")
        distribution_definition_schema = trac.load_schema(schemas, "distribution_definition_schema.csv")
        uk_monthly_scenarios_schema = trac.load_schema(schemas, "uk_monthly_scenarios_schema.csv")
        uk_macroeconomic_model_coefficients_schema = trac.load_schema(schemas, "uk_macroeconomic_model_coefficients_schema.csv")

        return {
            "sme_rating_master_scale": trac.ModelInputSchema(rating_master_scale_schema),
            "sme_transition_matrix": trac.ModelInputSchema(transition_matrix_schema),
            "uk_monthly_scenarios": trac.ModelInputSchema(uk_monthly_scenarios_schema),
            "uk_distribution_definition": trac.ModelInputSchema(distribution_definition_schema),
            "uk_macroeconomic_model_coefficients": trac.ModelInputSchema(uk_macroeconomic_model_coefficients_schema),
        }

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        uk_corporate_pd_term_structures_schema = trac.load_schema(schemas, "uk_corporate_pd_term_structures_schema.csv")

        return {
            "uk_sme_pd_term_structures": trac.define_output_table(uk_corporate_pd_term_structures_schema.table.fields, label="UK SME PD term structures"),
            }

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")
        n_years = ctx.get_parameter("n_years")

        # The definitions of each grade
        sme_rating_master_scale = ctx.get_pandas_table("sme_rating_master_scale")

        # The transition matrix
        # TODO Clean missing values
        sme_transition_matrix = ctx.get_pandas_table("sme_transition_matrix")

        uk_distribution_definition = ctx.get_pandas_table("uk_distribution_definition")
        uk_monthly_scenarios = ctx.get_pandas_table("uk_monthly_scenarios")
        uk_macroeconomic_model_coefficients = ctx.get_pandas_table("uk_macroeconomic_model_coefficients")

        total_months = n_years * 12

        scenarios = {
            "scenarios": {"monthly_scenarios": uk_monthly_scenarios},
            "totalMonths": total_months,
            "reporting_date": reporting_date,
            "scenario_Names": ['Base', 'Good', 'Better', 'Bad', 'Worse'],
            "macroModel": {"variablesNames": ['UK Current Account % GDP', 'UK Unemployment']}
        }

        scenarios = convert_to_dot_notation(scenarios)

        uk_monthly_scenarios.rename(columns={
            "UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP": "UK Current Account % GDP",
            "UK_UNEMPLOYMENT": "UK Unemployment"
        }, inplace=True)

        uk_macroeconomic_model_coefficients.rename(columns={
            "UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP": "UK Current Account % GDP",
            "UK_UNEMPLOYMENT": "UK Unemployment"
        }, inplace=True)

        median = uk_distribution_definition["MEDIAN"][0]

        lookup = {
            "UK Current Account % GDP": "UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP",
            "UK Unemployment": "UK_UNEMPLOYMENT"
        }

        macroImpacts = _calculate_macro_adjustment(
            monthly_scenarios=uk_monthly_scenarios,
            coefficients=uk_macroeconomic_model_coefficients,
            variable_names=scenarios.macroModel["variablesNames"],
            scenario_names=scenarios.scenario_Names,
            median=median,
            lookup=lookup
        )

        #############################################################################################
        #                               UK SME portfolio calculation                                #
        #############################################################################################

        logger.info(f"Generating the SME PD curves for a {reporting_date} reporting date")

        tm = TransitionMatrix(matrix=sme_transition_matrix, reporting_date=str(reporting_date), years=ceil(total_months / 12))
        external_ratings = tm.ratings.tolist()
        if "Default" in external_ratings:
            external_ratings.remove("Default")

        sme_pd_term_structures = _calculate_PiT_PDs(
            tm=tm.interpolated_term_structures,
            mrs=sme_rating_master_scale,
            external_ratings=external_ratings,
            internal_ratings=sme_rating_master_scale['Rating'].tolist(),
            scenario_Names=scenarios.scenario_Names,
            macroImpacts=macroImpacts
        )

        sme_pd_term_structures = calculate_weighted_average_curves(
            pd_term_structures=sme_pd_term_structures,
            masterRatingScale=sme_rating_master_scale,
            total_months=total_months
        )

        # Output the datasets
        ctx.put_pandas_table("uk_sme_pd_term_structures", sme_pd_term_structures)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "wholesale_ifrs9/config/generate_pd_curves_for_uk_sme_portfolio.yaml", "wholesale_ifrs9/config/sys_config.yaml")
