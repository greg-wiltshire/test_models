import typing as tp
from logging import Logger
from math import ceil
from typing import List
import numpy as np
import pandas as pd
# Load the TRAC runtime library
import tracdap.rt.api as trac
from tnp_ifrs9_ecl_calculation_engine.tnp_ifrs9_calculator_models.pd.TransitionMatrix import TransitionMatrix
from pandas.tseries.offsets import MonthEnd
from scipy.stats import norm
from scipy.stats import gumbel_r

# Load the schemas library
from trac_poc import schemas as schemas

"""
A model that generates monthly forecasts of PD term structures for the UK portfolios.
"""


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


def score_up_df(data: pd.DataFrame, variable_names: List[str], coefficients: pd.DataFrame, name: str, arg: int = 0) -> pd.DataFrame:
    """
    Calculate the score of some set of data points
    Args:
        arg: option to return just a column of scores
        name: column name for score values
        data (pd.DataFrame): a DataFrame with variables and their values, w or w/o column "name"
    Returns:
        pd.DataFrame with col "name" containing score values
    """

    data = data.copy()
    coefficients = coefficients.copy()

    if 'Intercept' in coefficients.columns:
        data[name] = data[variable_names].to_numpy() @ coefficients.drop(['Intercept'], axis=1).to_numpy().transpose() + coefficients["Intercept"][0]
    else:
        data[name] = data[variable_names].to_numpy() @ coefficients.to_numpy().transpose()
    if arg != 0:
        return data[name]
    else:
        return data


class convert_to_dot_notation(dict):
    """
    Access dictionary attributes via dot notation
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _calculate_macro_adjustment(monthly_scenarios: pd.DataFrame, coefficients: pd.DataFrame, variable_names: List[str], scenario_names: List[str], median: float, beta: float = 1):
    """
    Calculate scenario-specific macro-scalars from self.scenarioDefinitions.monthly_scenarios
    """

    lookup = {
        "UK Current Account % GDP": "UK_CURRENT_ACCOUNT_AS_PERCENTAGE_OF_GDP",
        "UK Unemployment": "UK_UNEMPLOYMENT"
    }

    sc = monthly_scenarios.copy()
    sc["Prediction"] = score_up_df(sc[lookup.keys()], variable_names, coefficients=coefficients, name="Prediction", arg=1)
    # Not sure why we needed to add this in
    sc["Prediction"] = sc["Prediction"].astype(float)
    # compute conditional pds
    sc['cond'] = norm.cdf(sc["Prediction"])
    likely_PD = norm.cdf(median)
    scalars = {}
    for s in scenario_names:
        _mm = sc[sc["Scenario"] == s][["Date", "Prediction", "Scenario", "Weight", "cond"]]
        prev_cum = 0
        prev_likely_cum = 0
        # recursive
        for index, row in _mm.iterrows():
            if index == 0:
                _mm.loc[index, 'Cumulative'] = _mm.loc[index, 'cond'] / 12
                _mm.loc[index, 'Survival'] = 1
                _mm.loc[index, 'Likely_Cum'] = likely_PD / 12
                _mm.loc[index, 'Likely_Survival'] = 1
            else:
                _mm.loc[index, 'Cumulative'] = ((1 - prev_cum) * (_mm.loc[index, 'cond'] / 12)) + prev_cum
                _mm.loc[index, 'Survival'] = 1 - prev_cum
                _mm.loc[index, 'Likely_Cum'] = ((1 - prev_likely_cum) * (likely_PD / 12)) + prev_likely_cum
                _mm.loc[index, 'Likely_Survival'] = 1 - prev_likely_cum
            prev_cum = _mm.loc[index, 'Cumulative']
            prev_likely_cum = _mm.loc[index, 'Likely_Cum']
        _mm['Macro_Scalar'] = 1 + ((((_mm['Cumulative'] / _mm['Likely_Cum']) * (
                _mm['Survival'] / _mm['Likely_Survival'])) - 1) * beta)
        scalars[s] = _mm[["Date", "Scenario", "Macro_Scalar", "Weight"]]

    return scalars


def _calculate_PiT_PDs(tm, mrs, internal_ratings, external_ratings, scenario_Names, macroImpacts):
    """
    Create self.pd_term_structures by applying self.macro_impact to self.termStructure.interpolated_term_structures
    and self.masterRatingScale, proceed to calculate_weighted_average_curves()
    """
    ts = tm
    mrs = mrs
    _ts = pd.DataFrame()
    _ts = ts[["Date"]].copy()
    same = False
    if check_if_equal(internal_ratings, external_ratings):
        same = True
    for grade_no, r in enumerate(internal_ratings):
        if mrs[mrs["Rating"] == r]["Mid_PD"].tolist()[0] == 1:
            continue
        if same:
            internal_grade = str(r).replace(" ", "_").replace("+", "plus").replace("-", "minus")
            external_grade = internal_grade
        else:
            external_grade = mrs[mrs["Rating"] == r]["External_Rating"].tolist()[0]
            internal_grade = str(r).replace(" ", "_").replace("+", "plus").replace("-", "minus")
        _ts[internal_grade + '_ts'] = ts[external_grade]
        _ts[internal_grade + "_ttc_pd"] = mrs[mrs["Rating"] == r]["Mid_PD"].tolist()[0]
    pd_term_structures = pd.DataFrame()
    for scenario_no, s in enumerate(scenario_Names):
        tmp = macroImpacts[s]
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        _ts["Date"] = pd.to_datetime(_ts["Date"]) + MonthEnd(0)
        _pd = pd.merge(tmp, _ts, on="Date")
        for r in internal_ratings:
            internal_grade = str(r).replace(" ", "_").replace("+", "plus").replace("-", "minus")
            if mrs[mrs["Rating"] == r]["Mid_PD"].tolist()[0] == 1:
                continue
            bflag = (_pd[internal_grade + "_ttc_pd"] >= 1).astype(int)
            _pd = _pd.copy()
            _pd[internal_grade + '_ttc_ts'] = bflag + (1 - bflag) * (_pd[internal_grade + "_ttc_pd"] * _pd[internal_grade + "_ts"])
            bflag = ((_pd[internal_grade + '_ttc_ts'] * _pd['Macro_Scalar']) >= 1).astype(int)
            _pd[internal_grade + '_pit_pd'] = bflag + (1 - bflag) * (_pd[internal_grade + '_ttc_ts'] * _pd['Macro_Scalar'])
            bflag = ((_pd[internal_grade + "_ttc_pd"] * _pd['Macro_Scalar']) >= 1).astype(int)
            _pd[internal_grade + '_pit_nts_pd'] = bflag + (1 - bflag) * (_pd[internal_grade + "_ttc_pd"] * _pd['Macro_Scalar'])
        if scenario_no == 0:
            pd_term_structures = _pd
        else:
            pd_term_structures = pd.concat([pd_term_structures, _pd], ignore_index=True)
    return pd_term_structures


def calculate_weighted_average_curves(pd_term_structures, masterRatingScale, total_months):
    """
    Calculate weighted pit pd curves, append to self.pd_term_structures

        NB: The specification of total_months is not intuitive
            To arrive at exactly 120 projections per scenario (incl. weighted)
            when the transition matrix rolled out ten years, one must specify 119 (i.e. as we also project for the observation month)
            It would be better for an input of 120 to be interpreted as 119

    """
    mrs = masterRatingScale
    temp = mrs[mrs["Mid_PD"].values != 1].iloc[:, 0].astype('string')
    temp = temp.replace(' ', '_', regex=True).replace('\-', 'minus', regex=True).replace('\+', 'plus', regex=True) + '_pit_pd'
    temp = list(temp)
    pit_pds = np.array(pd_term_structures[temp])
    w = np.array(pd_term_structures["Weight"])
    calc = pd.DataFrame(np.repeat(w, len(temp), 0).reshape(pd_term_structures.shape[0], len(temp))).mul(pit_pds)
    aggr = pd_term_structures[["Date"]].copy()
    # Implied so that scenario is dropped on the sum
    # aggr["Scenario"] = aggr["Scenario"].astype(object)
    aggr[temp] = calc
    aggr.index = aggr["Date"].to_list()
    aggr = aggr.resample("M").sum(numeric_only=True)
    aggr.insert(0, "Weight", 1)
    aggr.insert(0, "Scenario", 'Weighted')
    aggr.insert(0, "Date", aggr.index)
    aggr = aggr[~pd.isnull(aggr['Date'])]
    aggr = aggr.reset_index(drop=True)
    # fix on 2022.10.31 accounts for the case of total_months < length projections base case input, alignment with PiTPD.py
    aggr = aggr[aggr.index < total_months + 1]

    pd_term_structures = pd.concat([pd_term_structures, aggr], ignore_index=True)

    return pd_term_structures


def check_if_equal(list_1, list_2):
    """ Check if both the lists are of same length and if yes then compare
    sorted versions of both the list to check if both of them are equal
    i.e. contain similar elements with same frequency. """

    if len(list_1) != len(list_2):
        return False
    return sorted(list_1) == sorted(list_2)


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
        uk_monthly_scenarios_schema = trac.load_schema(schemas, "uk_monthly_scenarios_schema.csv")
        distribution_definition_schema = trac.load_schema(schemas, "distribution_definition_schema.csv")
        uk_macroeconomic_model_coefficients_schema = trac.load_schema(schemas, "uk_macroeconomic_model_coefficients_schema.csv")

        return {
            "corporate_rating_master_scale": trac.ModelInputSchema(rating_master_scale_schema),
            "corporate_transition_matrix": trac.ModelInputSchema(transition_matrix_schema),
            "uk_monthly_scenarios": trac.ModelInputSchema(uk_monthly_scenarios_schema),
            "uk_distribution_definition": trac.ModelInputSchema(distribution_definition_schema),
            "uk_macroeconomic_model_coefficients": trac.ModelInputSchema(uk_macroeconomic_model_coefficients_schema),
        }

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        uk_corporate_pd_term_structures_schema = trac.load_schema(schemas, "uk_corporate_pd_term_structures_schema.csv")

        return {
            "uk_corporate_pd_term_structures": trac.define_output_table(uk_corporate_pd_term_structures_schema.table.fields, label="UK corporate PD term structures"),
        }

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")
        n_years = ctx.get_parameter("n_years")

        logger.info(f"Generating the Corporate PD curves for a {reporting_date} reporting date")

        # The definitions of each grade
        corporate_rating_master_scale = ctx.get_pandas_table("corporate_rating_master_scale")

        # The transition matrix
        # TODO Clean missing values
        corporate_transition_matrix = ctx.get_pandas_table("corporate_transition_matrix")

        uk_distribution_definition = ctx.get_pandas_table("uk_distribution_definition")
        uk_monthly_scenarios = ctx.get_pandas_table("uk_monthly_scenarios")
        uk_macroeconomic_model_coefficients = ctx.get_pandas_table("uk_macroeconomic_model_coefficients")

        total_months = n_years * 12

        tm = TransitionMatrix(matrix=corporate_transition_matrix, reporting_date=str(reporting_date), years=ceil(total_months / 12))

        __external_ratings = tm.ratings.tolist()
        __external_ratings = __external_ratings.remove("Default")

        distribution = {
            "mu": uk_distribution_definition["MU"][0],
            "sigma": uk_distribution_definition["SIGMA"][0],
            "range_low": uk_distribution_definition["DISTRIBUTION_LOW_VALUE"][0],
            "range_high": uk_distribution_definition["DISTRIBUTION_HIGH_VALUE"][0],
            "range": uk_distribution_definition["DISTRIBUTION_RANGE"][0],
            "interval": uk_distribution_definition["DISTRIBUTION_INTERVAL"][0],
        }

        scenarios = {
            "scenarios": {"monthly_scenarios": uk_monthly_scenarios},
            "totalMonths": total_months,
            "reporting_date": reporting_date,
            "scenario_Names": ['Base', 'Good', 'Better', 'Bad', 'Worse'],
            "macroModel": {"variablesNames": ['UK Current Account % GDP', 'UK Unemployment']},
            "score_up_df": score_up_df
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

        median = gumbel_r.mean(distribution['mu'], distribution['sigma'])

        macroImpacts = _calculate_macro_adjustment(
            monthly_scenarios=uk_monthly_scenarios,
            coefficients=uk_macroeconomic_model_coefficients,
            variable_names=scenarios.macroModel["variablesNames"],
            scenario_names=scenarios.scenario_Names,
            median=median
        )

        pd_term_structures = _calculate_PiT_PDs(
            tm=tm.interpolated_term_structures,
            mrs=corporate_rating_master_scale,
            external_ratings=tm.ratings.tolist(),
            internal_ratings=corporate_rating_master_scale['Rating'].tolist(),
            scenario_Names=scenarios.scenario_Names,
            macroImpacts=macroImpacts
        )

        pd_term_structures = calculate_weighted_average_curves(pd_term_structures, masterRatingScale=corporate_rating_master_scale, total_months=total_months)

        # Output the datasets
        ctx.put_pandas_table("uk_corporate_pd_term_structures", pd_term_structures)
        # ctx.put_pandas_table("uk_corporate_matrix_multiplied", tm.matrix_multiplied)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "trac_poc/config/generate_corporate_pd_curve.yaml", "trac_poc/config/sys_config.yaml")
