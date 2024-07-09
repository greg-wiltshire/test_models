from logging import Logger
from typing import List, Dict

import numpy as np
import pandas as pd

from pandas._libs.tslibs.offsets import MonthEnd
from scipy.stats import norm


def score_up_df(data: pd.DataFrame, variable_names: List[str], coefficients: pd.DataFrame, name: str, arg: int = 0) -> pd.DataFrame:
    """
    Calculate the score of some set of data points
    Args:
        arg: option to return just a column of scores
        variable_names: column name for score values
        coefficients
        name
        data (pd.DataFrame): a DataFrame with variables and their values, w or w/o column "name"
    Returns:
        pd.DataFrame with col "name" containing score values
    """

    data = data.copy()
    coefficients = coefficients.copy()

    if 'Intercept' in coefficients.columns:
        result = data[variable_names].to_numpy() @ coefficients.drop(['Intercept'], axis=1).to_numpy().transpose() + coefficients["Intercept"][0]
    else:
        result = data[variable_names].to_numpy() @ coefficients.to_numpy().transpose()

    data[name] = list(result.flatten())

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


def _calculate_macro_adjustment(monthly_scenarios: pd.DataFrame, coefficients: pd.DataFrame, variable_names: List[str], scenario_names: List[str], median: float, lookup: Dict[str, str], beta: float = 1):
    """
    Calculate scenario-specific macro-scalars from self.scenarioDefinitions.monthly_scenarios
    """

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
    for column in temp:
        aggr[column] = aggr[column].astype(float)
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