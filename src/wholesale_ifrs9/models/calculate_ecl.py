import math
import typing as tp
from datetime import date
from typing import Dict
from typing import Tuple

import pandas as pd
# Load the TRAC runtime library
import tracdap.rt.api as trac
from numpy import arange
from scipy.stats import norm

# Load the schemas library
from trac_poc import schemas as schemas

"""
A model that generates an ECL forecast for each facility.
"""

from collections import namedtuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


# Discount and weight metrics
def discounting(lst, interest_rate, weights, end_date, stage=1, error_fix=True):
    new_list = np.empty([len(lst)], dtype=float)
    new_list_weighted = np.empty([len(lst)], dtype=float)

    if interest_rate != 0:
        rate = 1 / np.full([len(lst)], 1 + interest_rate / 12)
        time = np.arange(1, len(lst) + 1)

        if error_fix:
            if (
                    end_date is not None and end_date != ""
                    and not pd.isna(end_date)
                    and stage == 2
            ):
                last_time = time[-1] + (
                        time[-1]
                        * (datetime.strptime(end_date, "%d/%m/%Y").day / 30)
                )
                time[-1] = last_time

        else:
            if (
                    end_date is not None and end_date != ""
                    and not pd.isna(end_date)
                    and stage == 2
            ):
                rate[-1] = (
                        interest_rate
                        / 12
                        * (datetime.strptime(end_date, "%d/%m/%Y").day / 30)
                )

        discount = np.power(rate, time)
    else:
        discount = np.full([len(lst)], 1)

    new_list = np.prod(np.vstack([lst, discount]), axis=0)
    new_list_weighted = np.prod(np.vstack([new_list, weights]), axis=0)

    return (
        new_list,
        [sum(new_list)],
        new_list_weighted,
        [sum(new_list_weighted)],
    )


def calculate_recovery_rate(instrument_differentiator: str, no_collateral_recovery_rates):
    if instrument_differentiator == 'Sov':
        return no_collateral_recovery_rates["sov"]
    elif instrument_differentiator == 'Non-Sov-Loan':
        return no_collateral_recovery_rates["non_sov_loan"]
    else:
        return no_collateral_recovery_rates["non_sov_bond"]


def calculate_lgb_no_collateral(interest_rate: float, EAD: np.array, instrument_differentiator: str, pit_PD: list,
                                ttc_PD: float, correlation: float, stage: int, no_collateral_recovery_rates, time_of_recovery=None):
    if stage == 3:
        recovery_rate = calculate_recovery_rate(instrument_differentiator, no_collateral_recovery_rates)
        return np.full([1], EAD[0] * recovery_rate), \
            np.full([1], recovery_rate), \
            np.full([1], EAD[0] * recovery_rate), \
            np.full([1], recovery_rate), \
            np.full([1], 0)

    recovery_rate = calculate_recovery_rate(instrument_differentiator, no_collateral_recovery_rates)
    collateral = np.full([len(EAD)], 0)

    # TODO: ASSUMPTION CHECK-> PIT PDs are without Term structure effects as these are idiosyncratic risks,
    #  This looks for just a way of backing out macro effect from difference in PiT PDs and TTC PDs?
    _lgb = EAD - ((EAD * recovery_rate) * (1 + interest_rate) ** (-time_of_recovery / 12))
    _lgd = _lgb / EAD

    el = ttc_PD * _lgd
    k = (norm.ppf(ttc_PD) - norm.ppf(list(el))) / math.sqrt(1 - correlation)
    j = norm.ppf(pit_PD[:len(EAD)]) - k
    l = norm.cdf(j)
    lgd_adjusted = (l / pit_PD[:len(EAD)])
    lgb_adjusted = lgd_adjusted * EAD

    return _lgb, _lgd, lgb_adjusted, lgd_adjusted, collateral

def calculate_collateral_index(projections, switch_off=False):
    collateral_index = np.empty(len(projections), dtype=float)
    if not switch_off:

        for i, p in enumerate(projections):
            if i == 0:
                collateral_index[0] = 1
            else:
                collateral_index[i] = collateral_index[i-1] * (1+p/12)
    else:
        collateral_index = np.full([len(projections)], 1)

    return collateral_index


def calculate_lgb_other_collateral(interest_rate: float, EAD: list, collateral_valuation: float, collateral_type: str,
                                   last_ead=0, stage: int = 1, time_of_recovery=None, cost_of_recovery=None, LGD_floor=None, haircuts= None, projections=None):
    haircut = haircuts[collateral_type]

    # HERE
    collateral_index = calculate_collateral_index(projections)

    collateral_value = collateral_valuation * collateral_index[1:len(EAD) + 1]
    post_haircut_valuation_adj = collateral_value * (1 - haircut)
    post_haircut_valuation = collateral_valuation * (1 - haircut)

    discount_rate = (1 + interest_rate) ** (-time_of_recovery / 12)

    recovery_cost_adj = cost_of_recovery * post_haircut_valuation_adj
    recovery_cost = cost_of_recovery * post_haircut_valuation

    discounted_recovery_adj = (post_haircut_valuation_adj - recovery_cost_adj) * discount_rate
    discounted_recovery = (post_haircut_valuation - recovery_cost) * discount_rate

    loss_adj = np.maximum(np.subtract(EAD, discounted_recovery_adj), 0)
    loss = np.maximum(np.subtract(EAD, np.full([len(EAD)], discounted_recovery)), np.full([len(EAD)], 0))

    LGB_Adj = np.maximum(EAD * LGD_floor, loss_adj)
    LGB = np.maximum(EAD * LGD_floor, loss)
    LGD = LGB / EAD
    LGD_Adj = LGB_Adj / EAD

    return LGB, LGD, LGB_Adj, LGD_Adj, collateral_value

def calculate_PPD(LTV):
        # this code finds the key that is closest to the LTV and gets the value

        ppd_parameters = {0.0: 0.15, 0.45: 0.2, 0.6: 0.35, 0.75: 0.5, 2.0: 1.0}

        if LTV > list(ppd_parameters.keys())[-1]:
            return list(ppd_parameters.values())[-1]

        return [v for k, v in ppd_parameters.items() if k <= LTV][-1]

def calculate_lgb_property_collateral(latest_valuation: float, interest_rate: float, IEPE: bool, property_type: str,
                                      scenario: str, EAD: list, current_index: float, last_ead=0, cost_of_recovery=None, LGD_floor=None, haircuts= None, time_of_recovery_values=None, ppd_parameters=None, projections=None):

    if IEPE == True:
        haircut = haircuts[property_type][1]
    else:
        haircut = haircuts[property_type][0]

    time_of_recovery = time_of_recovery_values[scenario]

    collateral_index = calculate_collateral_index(projections)

    # predicted collateral value with and without future predicted indexation
    future_idx = current_index * collateral_index[1:len(EAD) + 1]
    predicted_valuation_adj = latest_valuation * future_idx
    predicted_valuation = latest_valuation * np.full([len(EAD)], current_index)

    # Forced sale value with and without indexation
    predicted_forced_sale_value_adj = predicted_valuation_adj * (1 - haircut)
    predicted_forced_sale_value = predicted_valuation * (1 - haircut)

    # Predicted recovery costs after haircut
    recovery_cost_adj = cost_of_recovery * predicted_forced_sale_value_adj
    recovery_cost = cost_of_recovery * predicted_forced_sale_value

    # discount factor
    discount_factor = (1 + interest_rate) ** (-(time_of_recovery / 12))

    # total recovery after recovery costs
    total_recovery_adj = predicted_forced_sale_value_adj - recovery_cost_adj
    total_recovery = predicted_forced_sale_value - recovery_cost

    # discounted recoveries
    discounted_recovery_adj = total_recovery_adj * discount_factor
    discounted_recovery = total_recovery * discount_factor

    # total loss after recovery and costs
    total_loss_adj = np.maximum(np.subtract(EAD, discounted_recovery_adj), np.full([len(EAD)], 0))
    total_loss = np.maximum(np.subtract(EAD, discounted_recovery), np.full([len(EAD)], 0))

    LTV = np.divide(EAD, predicted_valuation).tolist()
    LTV_adj = np.divide(EAD, predicted_valuation_adj).tolist()

    ppd = np.asarray(list(map(calculate_PPD, LTV)))
    ppd_adj = np.asarray(list(map(calculate_PPD, LTV_adj)))

    discounted_loss_ppd_adj = total_loss_adj * ppd_adj
    discounted_loss_ppd = total_loss * ppd

    # Loss (minimised to floor)
    LGB_adj = np.maximum(EAD * LGD_floor, discounted_loss_ppd_adj)
    LGB = np.maximum(EAD * LGD_floor, discounted_loss_ppd)

    LGD = LGB / EAD
    LGD_Adj = LGB_adj / EAD

    return LGB, LGD, LGB_adj, LGD_Adj, predicted_valuation_adj


# number of months between 2 dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


# as per Excel PMT function
def PMT(rate, nper, pv, fv=0, p_type=0):
    if rate != 0:
        pmt = (rate * (fv + pv * (1 + rate) ** nper)) / ((1 + rate * p_type) * (1 - (1 + rate) ** nper))
    else:
        pmt = (-1 * (fv + pv) / nper)
    return pmt


def IPMT(rate, per, nper, pv, fv=0, p_type=0):
    ipmt = -(((1 + rate) ** (per - 1)) * (pv * rate + PMT(rate, nper, pv, fv, p_type))
             - PMT(rate, nper, pv, fv, p_type))
    return ipmt


def PPMT(rate, per, nper, pv, fv=0, p_type=0):
    ppmt = PMT(rate, nper, pv, fv, p_type) \
           - IPMT(rate, per, nper, pv, fv, p_type)
    return ppmt


# revolving EAD is just CCF x headroom
def calculate_value_at_default_revolving(pv, ccf, limit):
    undrawn = limit - pv
    return max(pv + (undrawn * ccf), pv)


def cal_EAD_term_structure_amortising_3m_interest(interest_rate, start_date: str, end_date: str, payment_freq: int, pv,
                                                  bullet, pay_scale_list, reporting_date: str, report_end_date: str):
    # set up dates in correct formats and storage variables
    last_ead = 0
    # numpy arrays
    #EAD_with_dates = namedtuple('EAD_scenarios_with_dates', ('dates', 'EADs', 'last_ead'))

    s_date = datetime.strptime(start_date, '%d/%m/%Y')
    rpt_date = datetime.strptime(reporting_date, '%d/%m/%Y')
    e_date = datetime.strptime(report_end_date, '%d/%m/%Y')
    m_date = datetime.strptime(end_date, '%d/%m/%Y')

    payment_months = []

    # time in months to various events
    total_time = diff_month(e_date, s_date)
    time_to_rpt = diff_month(rpt_date, s_date)
    time_to_end = diff_month(e_date, rpt_date)
    time_to_maturity = diff_month(m_date, rpt_date)

    if time_to_end < 0:
        raise Exception("Calculation Error: Please ensure Report Date has not been changed")

    # get payment months dependent upon open date and payment frequency
    if payment_freq == 1:
        payment_months = [i for i in range(1, 13)]

    elif payment_freq == 3:
        for m in [0, 3, 6, 9]:
            payment_months.append((s_date + relativedelta(months=m)).month)

    elif payment_freq == 6:
        for m in [0, 6]:
            payment_months.append((s_date + relativedelta(months=m)).month)

    elif payment_freq == 12:
        payment_months = [s_date.month]

    # get future payment dates

    # TODO : numpy a range
    _dates = np.arange(np.datetime64(s_date + relativedelta(months=time_to_rpt + 1)),
                       np.datetime64(s_date + relativedelta(months=time_to_maturity + time_to_rpt + 1)),
                       np.timedelta64(1, 'M'), dtype='datetime64[M]') + np.timedelta64(s_date.day - 1, 'D')

    _use_date_all = _dates[:min(time_to_end, time_to_maturity)]
    _dates_mask = [(_dates.astype('datetime64[M]').astype(int) % 12 + 1 == x) for x in payment_months]
    ones = np.array(_dates_mask).astype(int)
    col_totals = ones.sum(axis=0)
    dates_mask = np.array(col_totals).astype(bool)
    dates_mask_use = dates_mask[:min(time_to_end, time_to_maturity)]

    dates = _dates[dates_mask]
    use_date = _use_date_all[dates_mask_use]
    bool_mask = np.array(dates_mask_use).astype(int)

    # calculate amortisation schedules
    # where there is less than 1 payment to maturity
    if len(dates) == 0:
        _paid = np.arange(np.datetime64(s_date + relativedelta(months=1)),
                          np.datetime64(s_date + relativedelta(months=time_to_rpt + 1)),
                          np.timedelta64(1, 'M'), dtype='datetime64[M]') + np.timedelta64(s_date.day - 1, 'D')

        _paid_mask = [(_paid.astype('datetime64[M]').astype(int) % 12 + 1 == x) for x in payment_months]
        _ones = np.array(_paid_mask).astype(int)
        _col_totals = _ones.sum(axis=0)
        paid_mask = np.array(_col_totals).astype(bool)
        paid = _paid[paid_mask]

        if len(paid) == 0:
            last_payment_date = e_date - relativedelta(months=payment_freq)
        else:
            last_payment_date = paid.astype(datetime)[-1]

        time_since_last_payment = (rpt_date - datetime.combine(last_payment_date, datetime.min.time())).days
        int_rate = interest_rate * (time_since_last_payment / 365)
        current_exposure = max(pv * (1 + int_rate), 0)
        amortisation = np.full([min(time_to_end, time_to_maturity)], current_exposure)
        interest = (((1 + (interest_rate / 12)) ** 3) - 1) * np.roll(amortisation, 1)
        interest[0] = (((1 + (interest_rate / 12)) ** 3) - 1) * current_exposure
        ead = np.maximum(amortisation + interest, np.full([len(amortisation)], 0))

        mask = ead > 0
        ead = ead[mask]
        _dates = _dates[mask]

        last_ead = ead[-1] * (s_date.day / 30)
        # HERE
        return _dates, ead, last_ead

    # Time 0 calculation
    last_payment_date = dates.astype(datetime)[0] - relativedelta(months=payment_freq)
    time_since_last_payment = (rpt_date - datetime.combine(last_payment_date, datetime.min.time())).days
    int_rate = interest_rate * (time_since_last_payment / 365)
    current_exposure = max(pv * (1 + int_rate), 0)

    # amortisation schedule no interest or prepayment
    amortisation = np.full([min(time_to_end, time_to_maturity)], current_exposure)
    pmts = np.cumsum(np.prod(np.vstack(
        [PPMT((payment_freq / 12) * interest_rate, np.cumsum(bool_mask), len(dates), amortisation[0], -bullet),
         bool_mask]), axis=0))
    amortisation_paid = np.maximum(amortisation + pmts, np.full([len(amortisation)], 0))

    # prepayments and interest
    _prepayments = np.array(pay_scale_list[time_to_rpt + 1:])
    if len(_prepayments) >= min(time_to_end, time_to_maturity):
        prepayments = _prepayments[0:min(time_to_end, time_to_maturity)]
    else:
        required = min(time_to_end, time_to_maturity) - len(_prepayments)
        final_pp_rate = np.full([required], pay_scale_list[120])
        prepayments = np.append(_prepayments, final_pp_rate)

    interest = (((1 + (interest_rate / 12)) ** 3) - 1) * np.roll(amortisation_paid, 1)
    prepayment = prepayments * np.roll(amortisation_paid, 1)
    interest[0] = (((1 + (interest_rate / 12)) ** 3) - 1) * current_exposure
    prepayment[0] = current_exposure * pay_scale_list[min(120, time_to_rpt)]

    amortisation2 = np.maximum(amortisation_paid - prepayment, np.full([len(amortisation_paid)], 0))
    ead = amortisation2 + interest

    mask = ead > 0
    ead = ead[mask]

    if len(ead) == 0:
        ead = np.full([1], 0.000001)
    else:
        _use_date_all = _use_date_all[mask]

    if time_to_end == time_to_maturity:
        last_ead = ead[-1] * (s_date.day / 30)
    else:
        last_ead = ead[-1]

    return _use_date_all, ead, last_ead


# TODO: Assumption check->
#   simple CCF calculation and no Prepayments used
def cal_EAD_term_structure_revolving(pv, reporting_date: str, ccf, bt, limit, stage):
    reporting_date_dt = datetime.strptime(reporting_date, '%d/%m/%Y')
    #EAD_with_dates = namedtuple('EAD_scenarios_with_dates', ('dates', 'EADs', 'last_ead'))
    val = calculate_value_at_default_revolving(pv, ccf, limit)

    if stage == 1:
        EAD_list_val = np.full([min(int(bt) + 1, 13)], val)
        _dates = np.arange(np.datetime64(reporting_date_dt + relativedelta(months=1)),
                           np.datetime64(reporting_date_dt + relativedelta(months=min(int(bt) + 1, 13))),
                           np.timedelta64(1, 'M'), dtype='datetime64[M]')
    else:
        EAD_list_val = np.full([int(bt)], val)
        _dates = np.arange(np.datetime64(reporting_date_dt + relativedelta(months=1)),
                           np.datetime64(reporting_date_dt + relativedelta(months=int(bt) + 1)),
                           np.timedelta64(1, 'M'), dtype='datetime64[M]')

    return _dates, EAD_list_val, val


def update_record(account_id, stage: int, date, exposure, probationary_file):
    probationary_file = delete_record(probationary_file, account_id)
    df = pd.DataFrame(data={'AccountID': [account_id],
                            'SICR_Date': [date.strftime("%d/%m/%Y")],
                            'Stage': [stage],
                            'StageExposure': [exposure],
                            'Reason': ['Automated']})
    return pd.concat([probationary_file, df])


def delete_record(probationary_file, account_id):
    return probationary_file[probationary_file['AccountID'] != account_id].copy()


def is_internal_risk_move_stage_2(internal_inception_origination, internal_inception_reporting, stage_2_dict, internal_matrix):
    internal_origination = internal_matrix.get(internal_inception_origination)
    internal_reporting = internal_matrix.get(internal_inception_reporting)

    if internal_origination is None or math.isnan(internal_origination):
        return internal_reporting >= int(stage_2_dict['internal threshold check none'])

    if internal_origination <= int(stage_2_dict['internal threshold 1']):
        return internal_reporting >= int(stage_2_dict['internal threshold check 1'])

    if internal_origination <= int(stage_2_dict['internal threshold 2']):
        return internal_reporting - internal_origination >= int(stage_2_dict['internal threshold check 2'])

    return internal_reporting - internal_origination >= int(stage_2_dict['internal threshold check default'])


# Gets the column and row coordinates
# grade = column
# notch = row
def get_external_risk_location(external_risk: str, external_matrix):
    #grade_notch = namedtuple('grade_notch', ('grade', 'notch'))
    v = np.where(external_matrix == external_risk)
    grade = v[0]
    notch = v[1]
    return grade, notch


# I have put the external risk in a matrix top row is highest grade
# Left most column is highest notch
# highest grade == index 0
# The last check is to see if we are above investment grade and that we
# not moving in a positive direction
def is_external_risk_move_stage_2(external_risk_grade_reporting, external_risk_grade_origination, stage_2_dict, staging_external_rating_mapping):
    # TODO: Assumption check->
    #   if external risk grade is missing (at inception or now), ignore this criteria
    if pd.isna(external_risk_grade_reporting) or pd.isna(external_risk_grade_origination):
        return False

    index_reporting = get_external_risk_location(external_risk_grade_reporting, staging_external_rating_mapping)
    index_old = get_external_risk_location(external_risk_grade_origination, staging_external_rating_mapping)
    index_investment = get_external_risk_location(stage_2_dict['investment grade external'], staging_external_rating_mapping)

    return index_reporting[0] - index_old[0] >= int(stage_2_dict['external grade drop']) or \
        (index_old[0] < index_investment[0] and \
         index_old[1] < index_investment[1] and \
         index_reporting[0] -index_old[0] <= 0 and \
         index_reporting[1]- index_old[1] >= int(stage_2_dict['external notch drop']))


def is_stage_2(account_id, dpd, other_stage_2_flag,
               external_risk_grade_reporting, external_risk_grade_origination,
               internal_inception_origination, internal_inception_reporting, exposure, internal_matrix, stage_2_dict, probationary_file, reporting_date, stage_2_probationary_period,
               staging_external_rating_mapping) -> Tuple[
    pd.DataFrame, bool]:
    # TODO: Assumption check->
    #   this is to deal with the option to use retail risk grades that are not in the MRS and don't have a
    #   separate Staging criteria
    internal_reporting = internal_matrix.get(internal_inception_reporting, "Not MRS")

    reporting_date_as_datetime = datetime.strptime(reporting_date, "%d/%m/%Y")

    if internal_reporting == "Not MRS":
        meets_stg2_criteria = dpd >= int(stage_2_dict['No. Days']) or \
                              other_stage_2_flag == 1
    else:

        meets_stg2_criteria = dpd >= int(stage_2_dict['No. Days']) or \
                              other_stage_2_flag == 1 or \
                              is_external_risk_move_stage_2(external_risk_grade_reporting, external_risk_grade_origination, stage_2_dict=stage_2_dict,
                                                            staging_external_rating_mapping=staging_external_rating_mapping) or \
                              is_internal_risk_move_stage_2(internal_inception_origination, internal_inception_reporting, stage_2_dict=stage_2_dict, internal_matrix=internal_matrix)

    # TODO Shouldn't this select the stage 2 flagged accounts?
    is_stage2 = probationary_file[probationary_file['AccountID'] == account_id]

    # TODO: Assumption check->
    #   probation if account has been in account less than 12 months (if so keep in stage 2, but keep initial date
    #   for future tracking
    if is_stage2.shape[0] > 0:
        stage_2_date = is_stage2['SICR_Date'].tolist()[0]
        previous_stage = is_stage2['Stage'].tolist()[0]

        # Do we go back to stage 2 on recovery?
        if previous_stage == 3:
            probationary_file = update_record(account_id, 2, reporting_date_as_datetime, exposure, probationary_file)
            return probationary_file, True

        time_in_stage_2_3 = relativedelta(reporting_date_as_datetime, datetime.strptime(stage_2_date, "%d/%m/%Y"))
        diff_in_months = time_in_stage_2_3.months + time_in_stage_2_3.years * 12

        if meets_stg2_criteria:
            # update date to latest date as to not start probationary process next time
            probationary_file = update_record(account_id, 2, reporting_date_as_datetime, exposure, probationary_file)
            return probationary_file, True

        if diff_in_months > stage_2_probationary_period:
            # probation is over, remove from file
            probationary_file = delete_record(probationary_file, account_id)
            return probationary_file, False
        else:
            # Does not meet criteria, but still in probationary period, don't change file
            return probationary_file, True

    # New into Stage 2
    if meets_stg2_criteria:
        # add to file
        probationary_file = update_record(account_id, 2, reporting_date_as_datetime, exposure, probationary_file)
        return probationary_file, True
    else:
        return probationary_file, False


def is_stage_3(account_id, dpd, other_stage_3_flag, internal_inception_reporting, exposure, internal_matrix, stage_3_dict, probationary_file, reporting_date):
    # TODO: Assumption check->
    #   this is to deal with the option to use retail risk grades that are not in the MRS and don't have a
    #   separate Staging criteria
    internal_reporting = internal_matrix.get(internal_inception_reporting, "Not MRS")

    reporting_date_as_datetime = datetime.strptime(reporting_date, "%d/%m/%Y")

    if internal_reporting == "Not MRS":
        meets_stg3_criteria = dpd >= stage_3_dict['No. Days'] or \
                              other_stage_3_flag == 1
    else:
        meets_stg3_criteria = dpd >= stage_3_dict['No. Days'] or \
                              other_stage_3_flag == 1 or \
                              internal_reporting >= stage_3_dict['internal risk']

    # Is it just in the probationary file or that their stage is stage 3 in the file
    is_stage3 = probationary_file[probationary_file['AccountID'] == account_id]

    current_stage = 1
    if is_stage3.shape[0] > 0:
        current_stage = is_stage3['Stage'].tolist()[0]

    if is_stage3.shape[0] > 0 and current_stage == 3:

        # Still Stage 3 - update date as last date in Stage 3
        if meets_stg3_criteria:
            probationary_file = update_record(account_id, 3, reporting_date_as_datetime, exposure, probationary_file)
            return probationary_file, True

        # probationary logic - move to Stage 2 - record dealt with in Stage 2 calculation
        return probationary_file, False

    else:
        # Move from Stage 1/2 to 3
        if meets_stg3_criteria:
            probationary_file = update_record(account_id, 3, reporting_date_as_datetime, exposure, probationary_file)
            return probationary_file, True
        else:
            return probationary_file, False


def calculate_ECL_stage_SICR_status(account_id, dpd, other_stage_2_flag, other_stage_3_flag,
                                    internal_inception_reporting, internal_inception_origination,
                                    external_risk_grade_reporting, external_risk_grade_origination, exposure,
                                    pd_model, mat: Dict[str, pd.DataFrame], stage_3_dict, probationary_file, reporting_date, stage_2_probationary_period,
                                    staging_external_rating_mapping, stage_2_dict):
    if pd_model in mat:
        mat = mat[pd_model]
    else:
        mat = mat['global']

    if 'Notch' not in mat.columns.values.tolist():
        mat['Notch'] = np.arange(len(mat))

    internal_matrix = dict(zip(list(mat.Rating), list(mat.Notch)))

    probationary_file, is_stage_3x = is_stage_3(
        account_id, dpd, other_stage_3_flag, internal_inception_reporting, exposure, internal_matrix,
        stage_3_dict, probationary_file=probationary_file, reporting_date=reporting_date)

    if is_stage_3x:
        return 3, 'Y', probationary_file

    probationary_file, is_stage_2x = is_stage_2(account_id, dpd, other_stage_2_flag,
                                                external_risk_grade_reporting, external_risk_grade_origination,
                                                internal_inception_origination, internal_inception_reporting, exposure, internal_matrix,
                                                stage_2_dict, probationary_file=probationary_file, reporting_date=reporting_date, stage_2_probationary_period=stage_2_probationary_period,
                                                staging_external_rating_mapping=staging_external_rating_mapping)

    if is_stage_2x:
        return 2, 'Y', probationary_file

    return 1, 'N', probationary_file


def CalculateECL(reporting_date: datetime.date, facilities: pd.DataFrame, ccf_df: pd.DataFrame, mat: Dict[str, pd.DataFrame],
                 stage_3_dict, probationary_file, stage_2_probationary_period,
                 staging_external_rating_mapping, scenarios, stage_2_dict, pd_curves, ead_prepayment_curves, revolving_function, amortisation_function,
                 use_curve, lgd_curve, lgd_correlations, no_collateral_recovery_rates,
                 cost_of_recovery_parameters,
                 lgd_floor_parameters,
                 time_to_recovery_parameters,
                 haircuts,
                 ppd_parameters,
                 projections
                 ):
    # df set up stores
    FUNCTION = "CalculateECL"
    df = pd.DataFrame()
    stage_keeper = {}
    ecl_keeper = {}
    stg2_ecl_keeper = {}
    stg1_ecl_keeper = {}

    # ensure data is in correct format

    if isinstance(reporting_date, datetime) or isinstance(reporting_date, date):
        reporting_date = reporting_date.strftime("%d/%m/%Y")
    else:
        try:
            a = datetime.strptime(reporting_date, "%d/%m/%Y")
        except:
            reporting_date = datetime.strptime(reporting_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    # get data size for progress reporting
    size = facilities.shape[0]
    cashflow_dfs = []
    scenario_dfs = []
    n_accounts = len(facilities)

    # iterator for progress reporting, could use enumerate, index not always same as row number
    # To future Greg this iterator cold cause an issue
    iterator = 0
    count = 0
    for index, facilities_row in facilities.iterrows():
        iterator += 1
        stg2_ecl = 0
        stg1_ecl = 0
        count += 1

        ECL_val_hold_dict = {}
        ECL_scenario_dist = {}
        account_ID = facilities_row["AccountID"]

        # Read facility data
        region = facilities_row["Region"]
        lgd_region = facilities_row["LGDRegion"]
        pdmodel = facilities_row["PDModel"]
        internalRatingPresent = facilities_row["InternalRatingPresent"]
        dpd = facilities_row["DaysPastDue"]
        collateral_type = facilities_row["CollateralType"]
        latest_valuation = facilities_row["Valuation"]
        interest_rate = facilities_row["InterestRate"]
        IEPE = facilities_row["IPRE"] == "IPRE Haircut"
        property_type = facilities_row["PropertyType"]
        instrument_differentiator = facilities_row[
            "InstrumentDifferentiator"
        ]
        index_val = facilities_row["IndexValuation"]
        strength = facilities_row["CollateralStrength"]
        end_date = facilities_row["MaturityDate"]
        payment_freq = facilities_row["PaymentFrequency"]
        exposure = facilities_row["Exposure"]
        limit = facilities_row["Limit"]
        product = facilities_row["Product"]
        start_date = facilities_row["OpenDate"]
        n_row = facilities_row["n"]

        if (
                pd.isna(facilities_row["BulletPayment"])
                or facilities_row["BulletPayment"] is None
        ):
            bullet = 0
        else:
            bullet = facilities_row["BulletPayment"]

        payment_type = facilities_row["PaymentType"]

        if payment_type == "Revolving":

            row = ccf_df.loc[ccf_df["Product"] == product].iloc[0]
            ccf = row["CCF"]
            bt = row["Behavioural Term (Months)"]

        # Calculate account stage
        actual_stage, SICR_status, probationary_file = calculate_ECL_stage_SICR_status(
                account_ID,
                dpd,
                facilities_row["OtherStage2Flag"],
                facilities_row["OtherStage3Flag"],
                internalRatingPresent,
                facilities_row["InternalRatingOrigination"],
                facilities_row["ExternalRatingPresent"],
                facilities_row["ExternalRatingOrigination"],
                facilities_row["Exposure"],
                pdmodel,
                mat,
                stage_3_dict,
                probationary_file=probationary_file,
                reporting_date=reporting_date,
                stage_2_probationary_period=stage_2_probationary_period,
                staging_external_rating_mapping=staging_external_rating_mapping,
                stage_2_dict=stage_2_dict

        )

        # Adjustment for calculating Stage 2 ECL on all accounts.
        if actual_stage == 3:
            ECL_stage = 3
        else:
            ECL_stage = 2

        # Store stage for later
        stage_keeper[account_ID] = actual_stage
        ECL_scenario_dist["AccountID"] = account_ID
        ECL_scenario_dist["Stage"] = actual_stage
        ECL_scenario_dist["Region"] = region
        ECL_scenario_dist["PDModel"] = pdmodel
        ECL_scenario_dist["LGDRegion"] = lgd_region
        ECL_scenario_dist["n"] = n_row

        # calculate length of ECL calculation, 12 months for S1, maturity date for S2,
        # just using current info for stage 3
        if actual_stage == 1:
            report_end_date = (
                    datetime.strptime(reporting_date, "%d/%m/%Y")
                    + relativedelta(months=+12)
            ).strftime("%d/%m/%Y")
        elif actual_stage == 2:
            report_end_date = end_date
        elif actual_stage == 3:
            report_end_date = (
                    datetime.strptime(reporting_date, "%d/%m/%Y")
                    + relativedelta(months=+1)
            ).strftime("%d/%m/%Y")

        if actual_stage in [1, 2]:
            stage_2_report_end_date = end_date
        else:
            stage_2_report_end_date = report_end_date

        # add back base to scenarios as not in config file
        base_df = pd.DataFrame(
            {"Name": ["Base"], "Type": ["Base"], "OneInX": [1]}
        )
        # scenario_df = pd.concat([base_df,self.scenarios[region] ])

        scenario_df = base_df.append(
            scenarios[region], ignore_index=True
        )

        # TTC curve and column names not scenario dependent
        pd_curves_region = pd_curves[region + "_" + pdmodel]

        pit_pd_nts_column_name = None
        if ECL_stage != 3:
            ttc_column_name = (
                    internalRatingPresent.replace(" ", "_")
                    .replace("-", "minus")
                    .replace("+", "plus")
                    + "_ttc_pd"
            )
            internal_rating_column_name = (
                    internalRatingPresent.replace(" ", "_")
                    .replace("-", "minus")
                    .replace("+", "plus")
                    + "_pit_pd"
            )

            ttc_ts_column_name = (
                    internalRatingPresent.replace(" ", "_")
                    .replace("-", "minus")
                    .replace("+", "plus")
                    + "_ttc_ts"
            )
            ttc_pd = (
                (pd_curves_region[pd_curves_region["Scenario"] == "Base"])[
                    ttc_column_name
                ]
            ).values[0]
            pit_pd_nts_column_name = (
                    internalRatingPresent.replace(" ", "_")
                    .replace("-", "minus")
                    .replace("+", "plus")
                    + "_pit_nts_pd"
            )
        else:
            ttc_pd = 1

        # run all EAD amortisation schedules.... don't repeat calculation
        EAD_scenarios = {}
        # get prepayment curves
        us = ead_prepayment_curves["Upside"].to_numpy(dtype=float)
        ds = ead_prepayment_curves["Downside"].to_numpy(dtype=float)
        bs = ead_prepayment_curves["Base"].to_numpy(dtype=float)

        s_list = []
        # to save time calculating multiple amortisation schedules, check that prepayment curves are different,
        # if not just run required number of calculations
        if np.array_equal(us, bs) and np.array_equal(us, ds):
            s_list = ["Base"]
        elif np.array_equal(us, bs) and not np.array_equal(us, ds):
            s_list = ["Base", "Downside"]
        elif not np.array_equal(us, bs) and np.array_equal(us, ds):
            s_list = ["Base", "Upside"]
        elif not np.array_equal(us, bs) and not np.array_equal(us, ds):
            s_list = ["Upside", "Downside", "Base"]

        EAD_term_structure = None
        if ECL_stage != 3:
            for st in s_list:
                # retrieve prepayment curve for the scenario type (Upside, Downside, Base)
                prepayment_curve = ead_prepayment_curves[st]

                if payment_type == "Amortising":
                        # Amortisation schedule
                        EAD_term_structure = amortisation_function(
                            interest_rate,
                            start_date,
                            end_date,
                            int(payment_freq),
                            exposure,
                            bullet,
                            prepayment_curve,
                            reporting_date,
                            stage_2_report_end_date,
                        )

                elif payment_type == "Revolving":
                        # CCF
                        EAD_term_structure = revolving_function(
                            exposure,
                            reporting_date,
                            ccf,
                            bt,
                            limit,
                            ECL_stage,
                        )

                EAD_scenarios[st] = EAD_term_structure

            # As calculation is not always repeated, ensure all EAD profiles are captured for each scenarios
            if np.array_equal(us, bs) and np.array_equal(us, ds):
                EAD_scenarios["Downside"] = EAD_scenarios["Base"]
                EAD_scenarios["Upside"] = EAD_scenarios["Base"]
            elif np.array_equal(us, bs) and not np.array_equal(us, ds):
                EAD_scenarios["Upside"] = EAD_scenarios["Base"]
            elif not np.array_equal(us, bs) and np.array_equal(us, ds):
                EAD_scenarios["Downside"] = EAD_scenarios["Upside"]
        else:
            # Stage 3, set EAD schedule to current balance only, but ensure same format as STage 1 and Stage 2
            # EAD_with_dates = namedtuple(
            #     "EAD_scenarios_with_dates", ("dates", "EADs", "last_ead")
            # )
            dates = [datetime.strptime(reporting_date, "%d/%m/%Y")]
            EAD_scenarios["Downside"] = (
                dates, np.array([exposure]), exposure
            )
            EAD_scenarios["Upside"] = (
                dates, np.array([exposure]), exposure
            )
            EAD_scenarios["Base"] = (
                dates, np.array([exposure]), exposure
            )

        # calculate metrics for each scenario
        for index2, scenarios_row in scenario_df.iterrows():
            if scenarios_row["Use"] == False:
                continue

            scenarios_name = scenarios_row["Name"]
            scenarios_type = scenarios_row["Type"]
            one_in_X = scenarios_row["OneInX"]

            # retrieve EADs
            EAD = EAD_scenarios[scenarios_type][1]
            final_ead = EAD_scenarios[scenarios_type][2]

            # get current region, model and internal grade to get PD curve

            pd_no_ts = None
            ttc_pd_curve = None

            if ECL_stage != 3:
                pd_no_ts = (
                               (
                                   pd_curves_region[
                                       pd_curves_region["Scenario"] == scenarios_name
                                       ]
                               )[pit_pd_nts_column_name]
                           ).to_numpy(dtype=float)[0: EAD.shape[0]]
                pd_curve = (
                               (
                                   pd_curves_region[
                                       pd_curves_region["Scenario"] == scenarios_name
                                       ]
                               )[internal_rating_column_name]
                           ).to_numpy(dtype=float)[0: EAD.shape[0]]

                ttc_ts_curve = (
                                   (
                                       pd_curves_region[
                                           pd_curves_region["Scenario"] == scenarios_name
                                           ]
                                   )[ttc_ts_column_name]
                               ).to_numpy(dtype=float)[0: EAD.shape[0]]

                ttc_pd_curve = np.full([len(pd_no_ts)], ttc_pd)

            # retrieve LGD Engines

            if region != lgd_region:
                if one_in_X == 1:
                    pass
                else:
                    lgd_scenarios = scenarios[lgd_region]
                    lgd_scenario_name = lgd_scenarios[
                        (lgd_scenarios["Type"] == scenarios_type)
                        & (lgd_scenarios["OneInX"] == one_in_X)
                        ]["Name"].values[0]
            else:
                pass

            lgd_correlation = lgd_correlations[region + "_" + pdmodel]

            # calculate LGD through relevant Engine - returns tuple of LGDs:
            # LGD[0] = TTC Loss Value
            # LGD[1] = PiT Loss Value
            # LGD[2] = TTC LGD
            # LGD[3] = PiT LGD (Only this is used in the ECL calculation)
            # LGD[4] = Collateral Values

            if collateral_type == "Property":
                    LGD = calculate_lgb_property_collateral(
                        latest_valuation,
                        interest_rate,
                        IEPE,
                        property_type,
                        scenarios_type,
                        EAD,
                        index_val,
                        cost_of_recovery=cost_of_recovery_parameters["property"],
                        LGD_floor=lgd_floor_parameters["property"],
                        haircuts=haircuts["property"],
                        time_of_recovery_values = time_to_recovery_parameters["property"],
                        ppd_parameters=ppd_parameters,
                        projections=projections
                    )
            elif collateral_type == "Collateral":
                    LGD = calculate_lgb_other_collateral(
                        interest_rate, EAD, latest_valuation, strength, time_of_recovery=time_to_recovery_parameters["other_collateral"],
                        cost_of_recovery=cost_of_recovery_parameters["other_collateral"],
                        LGD_floor=lgd_floor_parameters["other_collateral"],
                        haircuts=haircuts["other_collateral"],
                        projections=projections
                    )
            else:
                    # need to update this line to pass in TTC_PD if TTC LGD
                if ECL_stage != 3:
                    if lgd_curve == "TTC":
                            LGD = calculate_lgb_no_collateral(
                                interest_rate,
                                EAD,
                                instrument_differentiator,
                                ttc_pd_curve,
                                ttc_pd,
                                lgd_correlation,
                                ECL_stage,
                                no_collateral_recovery_rates=no_collateral_recovery_rates,
                                time_of_recovery=time_to_recovery_parameters["no_collateral"]
                            )
                    else:
                            LGD = calculate_lgb_no_collateral(
                                interest_rate,
                                EAD,
                                instrument_differentiator,
                                pd_no_ts,
                                ttc_pd,
                                lgd_correlation,
                                ECL_stage,
                                no_collateral_recovery_rates=no_collateral_recovery_rates,
                                time_of_recovery=time_to_recovery_parameters["no_collateral"]
                            )
                else:
                    if lgd_curve == "TTC":
                            LGD = calculate_lgb_no_collateral(
                                interest_rate,
                                EAD,
                                instrument_differentiator,
                                [1],
                                ttc_pd,
                                lgd_correlation,
                                ECL_stage,
                                no_collateral_recovery_rates=no_collateral_recovery_rates,
                                time_of_recovery=time_to_recovery_parameters["no_collateral"]
                            )
                    else:
                            LGD = calculate_lgb_no_collateral(
                                interest_rate,
                                EAD,
                                instrument_differentiator,
                                [1],
                                # TODO this looks wrong
                                ttc_pd,
                                lgd_correlation,
                                ECL_stage,
                                no_collateral_recovery_rates=no_collateral_recovery_rates,
                                time_of_recovery=time_to_recovery_parameters["no_collateral"]
                            )

            # Choose which PD curve to apply based upon input
            if ECL_stage != 3:
                if use_curve == 3:
                    pd_curve_to_apply = pd_curve
                elif use_curve == 1:
                    pd_curve_to_apply = ttc_pd_curve
                elif use_curve == 2:
                    pd_curve_to_apply = ttc_ts_curve
                else:
                    pd_curve_to_apply = pd_no_ts

            # Get scenario weight
            weights = (
                          (
                              pd_curves_region[
                                  pd_curves_region["Scenario"] == scenarios_name
                                  ]
                          )["Weight"]
                      ).to_numpy(dtype=float)[0: EAD.shape[0]]
            scenarios_type = scenarios_row["Type"]

            # replace last ead (used for lGD) to the scaled EAD for ECL
            # Note, last EAD is scaled by number of days in maturity month (maturity date/30) so that ECL in last
            # month is scaled appropriately, but LGD needed actual EAD
            EAD_for_ECL = np.copy(EAD_scenarios[scenarios_type][1])
            EAD_for_ECL[-1] = final_ead

            # ECL Calculation - no discounting
            try:
                rpt_date = datetime.strptime(reporting_date, "%d/%m/%Y")

                if payment_type == "Revolving":
                    time_to_maturity = bt
                else:
                    m_date = datetime.strptime(end_date, "%d/%m/%Y")
                    time_to_maturity = diff_month(m_date, rpt_date)

                if ECL_stage == 3:
                    arrays = [EAD_for_ECL, LGD[3]]
                    Stage2ECL = np.prod(np.vstack(arrays), axis=0)
                    Stage1ECL = np.prod(np.vstack(arrays), axis=0)
                else:
                    arrays2 = [EAD_for_ECL, LGD[3], pd_curve_to_apply / 12]
                    Stage2ECL = np.prod(np.vstack(arrays2), axis=0)
                    arrays = [
                        EAD_for_ECL[:12],
                        LGD[3][:12],
                        pd_curve_to_apply[:12] / 12,
                    ]
                    Stage1ECL = np.prod(np.vstack(arrays), axis=0)

                if actual_stage in [2, 3] or time_to_maturity <= 12:
                    ECL = Stage2ECL
                    discount_stage = 2
                else:
                    ECL = Stage1ECL
                    discount_stage = 1

            except Exception as e:
                raise Exception(
                    "Error in calculating ECL for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )

            array_size = len(ECL)

            # set up result storage dictionaries to create DataFrame once calculation has completed
            if not ("AccountID" in ECL_val_hold_dict):
                ECL_val_hold_dict["AccountID"] = np.full(
                    [array_size], account_ID
                )

            if not ("Date" in ECL_val_hold_dict):
                ECL_val_hold_dict["Date"] = pd.to_datetime(
                    EAD_scenarios[scenarios_type][0][:array_size]
                )


            # Calculate ECL, discounted ECL, ECL x weight and Discounted ECL x weight for each scenario
            if ECL_stage != 3:
                (
                        ECL_discounted,
                        ECL_discounted_sum,
                        ECL_discounted_w,
                        ECL_discounted_sum_w,
                ) = discounting(
                        ECL,
                        interest_rate,
                        weights[:array_size],
                        end_date,
                        actual_stage,
                )
            else:
                (
                        ECL_discounted,
                        ECL_discounted_sum,
                        ECL_discounted_w,
                        ECL_discounted_sum_w,
                ) = discounting(ECL, 0, weights, end_date)

            (
                    Stage2_ECL_discounted,
                    Stage2_ECL_discounted_sum,
                    Stage2_ECL_discounted_w,
                    Stage2_ECL_discounted_sum_w,
            ) = discounting(
                    Stage2ECL,
                    interest_rate,
                    weights,
                    end_date,
                    discount_stage,
            )
            (
                    Stage1_ECL_discounted,
                    Stage1_ECL_discounted_sum,
                    Stage1_ECL_discounted_w,
                    Stage1_ECL_discounted_sum_w,
            ) = discounting(
                    Stage1ECL,
                    interest_rate,
                    weights[:12],
                    end_date,
                    discount_stage,
            )

            stg2_ecl += Stage2_ECL_discounted_sum_w[0]
            stg1_ecl += Stage1_ECL_discounted_sum_w[0]


            # Calculate LGD, LGD x weight for each scenario (LGD[3] = Pit LGD)
            # (uses discounting function, but uses 0 as rate as not to discount, discounted variables not used)

            try:
                (
                    LGD_Adj_discounted,
                    LGD_Adj_discounted_sum,
                    LGD_Adj_discounted_w,
                    LGD_Adj_discounted_sum_w,
                ) = discounting(LGD[3], 0, weights, end_date)
            except Exception as e:
                raise Exception(
                    "Error in calculating weighted PiT LGD for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )

            # Calculate EAD, EAD x weight for each scenario
            # (uses discounting function, but uses 0 as rate as not to discount, discounted variables not used)
            try:
                (
                    EAD_discounted,
                    EAD_discounted_sum,
                    EAD_discounted_w,
                    EAD_discounted_sum_w,
                ) = discounting(EAD_for_ECL, 0, weights, end_date)
                (
                    EAD_discounted2,
                    EAD_discounted_sum2,
                    EAD_discounted_w2,
                    EAD_discounted_sum_w2,
                ) = discounting(
                    EAD_scenarios[scenarios_type][1],
                    0,
                    weights,
                    end_date,
                )
            except Exception as e:
                raise Exception(
                    "Error in calculating weighted EAD for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )

            # Calculate LGD, LGD x weight for each scenario (LGD[1] = PiT Loss Value)
            # (uses discounting function, but uses 0 as rate as not to discount, discounted variables not used)

            try:
                (
                    LGD_discounted,
                    LGD_discounted_sum,
                    LGD_discounted_w,
                    LGD_discounted_sum_w,
                ) = discounting(LGD[1], 0, weights, end_date)
            except Exception as e:
                raise Exception(
                    "Error in calculating weighted TTC LGD for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )

            # Calculate Collateral Value for each scenario (LGD[4] = Collateral Value)
            # (uses discounting function, but uses 0 as rate as not to discount, discounted and weighted
            # variables not used)

            try:
                (
                    collateral_discounted,
                    collateral_discounted_sum,
                    collateral_discounted_w,
                    collateral_discounted_sum_w,
                ) = discounting(LGD[4], 0, weights, end_date)
            except Exception as e:
                raise Exception(
                    "Error in calculating weighted collateral for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )

            try:
                if ECL_stage != 3:
                    # Calculate PD, PD x weight for each scenario
                    # (uses discounting function, but uses 0 as rate as not to discount,
                    # discounted variables not used)
                    (
                        PD_discounted,
                        PD_discounted_sum,
                        PD_discounted_w,
                        PD_discounted_sum_w,
                    ) = discounting(
                        pd_curve_to_apply, 0, weights, end_date
                    )
                else:
                    # PDs set to 1 for Stage 3
                    # (uses discounting function to get appropriate weightings for each scenario)
                    (
                        PD_discounted,
                        PD_discounted_sum,
                        PD_discounted_w,
                        PD_discounted_sum_w,
                    ) = discounting(
                        np.full([len(ECL_discounted)], 1),
                        0,
                        weights,
                        end_date,
                    )
            except Exception as e:
                raise Exception(
                    "Error in calculating weighted PD for ID "
                    + str(account_ID)
                    + "("
                    + str(e)
                    + ")"
                )
            # Cash flows

            if index2 == 0:
                tag = "Base"
            else:
                tag = str(index2)

            # Add metrics to dictionary
            ECL_val_hold_dict[tag + "_LGD_Adj"] = LGD_Adj_discounted[
                                                  :array_size
                                                  ]
            ECL_val_hold_dict[
                tag + "_LGD_Adj_weighted"
                ] = LGD_Adj_discounted_w[:array_size]
            ECL_val_hold_dict[tag + "_LGD"] = LGD_discounted[:array_size]
            ECL_val_hold_dict[tag + "_LGD_weighted"] = LGD_discounted_w[
                                                       :array_size
                                                       ]
            ECL_val_hold_dict[tag + "_Collateral"] = collateral_discounted[
                                                     :array_size
                                                     ]
            ECL_val_hold_dict[
                tag + "_Collateral_weighted"
                ] = collateral_discounted_w[:array_size]
            ECL_val_hold_dict[tag + "_EAD"] = EAD_discounted2[:array_size]
            ECL_val_hold_dict[tag + "_EAD_weighted"] = EAD_discounted_w2[
                                                       :array_size
                                                       ]
            ECL_val_hold_dict[tag + "_ECL_EAD"] = EAD_discounted[
                                                  :array_size
                                                  ]
            ECL_val_hold_dict[tag + "_ECL_EAD_weighted"] = EAD_discounted[
                                                           :array_size
                                                           ]
            ECL_val_hold_dict[tag + "_ECL"] = ECL_discounted[:array_size]
            ECL_val_hold_dict[tag + "_ECL_weighted"] = ECL_discounted_w[
                                                       :array_size
                                                       ]
            ECL_val_hold_dict[tag + "_PD"] = PD_discounted[:array_size]
            ECL_val_hold_dict[tag + "_PD_weighted"] = PD_discounted_w[
                                                      :array_size
                                                      ]
            ECL_val_hold_dict[tag + "_Weight"] = weights[:array_size]

            ECL_scenario_dist[tag + "_ECL"] = ECL_discounted_sum[
                                              :array_size
                                              ]
            ECL_scenario_dist[tag + "_Collateral"] = collateral_discounted[
                                                     :array_size
                                                     ].mean()

            if ECL_stage != 3:
                survival = np.cumprod(1 - PD_discounted[:array_size] / 12)
                if array_size >= 12:
                    ECL_scenario_dist[tag + "_PD"] = [1 - survival[11]]
                else:
                    ECL_scenario_dist[tag + "_PD"] = [
                        ((1 - survival[array_size - 1]) / array_size) * 12
                    ]
            else:
                ECL_scenario_dist[tag + "_PD"] = [1]

            arrays2 = [
                EAD_discounted[:array_size],
                LGD_Adj_discounted[:array_size],
            ]
            lifetime_lgd_test = np.cumsum(
                np.prod(np.vstack(arrays2), axis=0)
            )
            total_ead = sum(EAD_discounted[:array_size])
            ECL_scenario_dist[tag + "_LGD"] = [
                lifetime_lgd_test[-1] / total_ead
            ]

            # For each scenario, create a total metric (sum of weighted metric across each scenario)

        try:
            for c in [
                "_LGD_Adj_weighted",
                "_LGD_weighted",
                "_EAD_weighted",
                "_ECL_weighted",
                "_PD_weighted",
                "_Collateral_weighted",
            ]:
                var_list = []
                for idx, account in enumerate(
                        ECL_val_hold_dict["AccountID"]
                ):
                    var = 0
                    for index2, scenarios_row in scenario_df.iterrows():
                        if scenarios_row["Use"] == False:
                            continue

                        scenarios_name = scenarios_row["Name"]
                        if index2 == 0:
                            var += ECL_val_hold_dict["Base" + c][idx]
                        else:
                            var += ECL_val_hold_dict[str(index2) + c][idx]

                    var_list.append(var)

                ECL_val_hold_dict[
                    "Total" + str(c).replace("_weighted", "")
                    ] = var_list
        except Exception as e:
            raise Exception(
                "Error in calculating total level metrics for "
                + str(account_ID)
                + "("
                + str(e)
                + ")"
            )

        print(str(account_ID))

        # Store ECL for later
        ecl_keeper[account_ID] = sum(ECL_val_hold_dict["Total_ECL"])
        stg2_ecl_keeper[account_ID] = stg2_ecl
        stg1_ecl_keeper[account_ID] = stg1_ecl

        cashflow_dfs.append(pd.DataFrame(ECL_val_hold_dict))
        scenario_dfs.append(pd.DataFrame(ECL_scenario_dist))

    df = pd.concat(cashflow_dfs)
    df2 = pd.concat(scenario_dfs)

    df2["weighted_ECL"] = np.nan
    df2["weighted_PD"] = np.nan
    df2["weighted_LGD"] = np.nan
    df2["weighted_Collateral"] = np.nan

    # Create Facility Level dataframe with results
    facility_level = facilities
    facility_level["Stage"] = np.nan
    facility_level["1-year PD"] = np.nan
    facility_level["Lifetime PD"] = np.nan
    facility_level["Average LGD"] = np.nan
    facility_level["Average LGD Adjusted"] = np.nan
    facility_level["LGD 0"] = np.nan
    facility_level["ECL"] = np.nan
    facility_level["Backbook"] = np.nan
    facility_level["Age"] = np.nan
    facility_level["Term"] = np.nan
    facility_level["Loss"] = np.nan
    facility_level["Loss Adjusted"] = np.nan
    facility_level["Total Exposure"] = np.nan
    facility_level["Coverage"] = np.nan

    current_date = datetime.strptime(reporting_date, "%d/%m/%Y")
    iterator = 0
    for index, facilities_row in facility_level.iterrows():
        iterator += 1
        account_ID = facilities_row["AccountID"]
        print(account_ID)

        try:
            # add facility level information
            stage = stage_keeper[account_ID]
            ecl = ecl_keeper[account_ID]
            stg2_ecl = stg2_ecl_keeper[account_ID]
            stg1_ecl = stg1_ecl_keeper[account_ID]
            internalRatingPresent = facilities_row["InternalRatingPresent"]
            region = facilities_row["Region"]
            pdmodel = facilities_row["PDModel"]
            product = facilities_row["Product"]
            payment_type = facilities_row["PaymentType"]

            if stage != 3:
                pd_curves_region = pd_curves[region + "_" + pdmodel]
                ttc_column_name = (
                        internalRatingPresent.replace(" ", "_")
                        .replace("-", "minus")
                        .replace("+", "plus")
                        + "_ttc_pd"
                )
                ttc_pd = (
                    (
                        pd_curves_region[
                            pd_curves_region["Scenario"] == "Base"
                            ]
                    )[ttc_column_name]
                ).values[0]
                facility_level.loc[index, "TTC_PD"] = ttc_pd
            else:
                facility_level.loc[index, "TTC_PD"] = 1

            if payment_type == "Revolving":
                row = ccf_df.loc[ccf_df["Product"] == product].iloc[0]
                bt = row["Behavioural Term (Months)"]
            else:
                bt = 0

            facility_level.loc[index, "Stage"] = stage
            facility_level.loc[index, "ECL"] = np.round(ecl, 2)
            facility_level.loc[index, "Stage2_ECL"] = np.round(stg2_ecl, 2)
            facility_level.loc[index, "Stage1_ECL"] = np.round(stg1_ecl, 2)

            facility_level.loc[index, "Coverage"] = np.round(
                ecl / facilities_row["Exposure"], 4
            )
            facility_level.loc[index, "Behavioural Term"] = bt

            # add ECL metrics to facility level data
            cashflows = df[df["AccountID"] == account_ID]
            facility_level.loc[index, "LGD 0"] = np.round(
                cashflows["Total_LGD_Adj"].values[0], 4
            )
            pd_df = cashflows["Total_PD"].to_numpy(dtype=float)
            lgd1_df = cashflows["Total_LGD"].to_numpy(dtype=float)
            lgd2_df = cashflows["Total_LGD_Adj"].to_numpy(dtype=float)
            ead_df = cashflows["Total_EAD"].to_numpy(dtype=float)
            collateral = cashflows["Total_Collateral"].to_numpy(
                dtype=float
            )

            # Calculate Lifetime LGD and Lifetime/12m PDs (conditional) for reporting and use in projections
            ead_total = sum(ead_df)
            survival = np.cumprod(1 - pd_df / 12)
            arrays = [ead_df, lgd1_df]
            arrays2 = [ead_df, lgd2_df]
            lifetime_loss = np.cumsum(np.prod(np.vstack(arrays), axis=0))
            lifetime_loss_adj = np.cumsum(
                np.prod(np.vstack(arrays2), axis=0)
            )
            loss0 = (
                    cashflows["Total_LGD_Adj"].values[0]
                    * facilities_row["Exposure"]
            )
            lifetime_pd = 1 - survival[-1]
            lifetime_lgd = lifetime_loss[-1] / ead_total
            lifetime_lgd_adj = lifetime_loss_adj[-1] / ead_total

            if stage == 3:
                annualised_pd = 1
                lifetime_pd = 1
                running_total_1y = 0
            else:
                if len(pd_df) >= 12:
                    annualised_pd = 1 - survival[11]
                elif len(pd_df) == 1:
                    annualised_pd = (1 - survival[0]) * 12
                else:
                    annualised_pd = (
                                            (1 - survival[len(pd_df) - 1]) / (len(pd_df))
                                    ) * 12

            # add PD / LGD metrics to facility level data
            facility_level.loc[index, "Annualised 1-year PD"] = np.round(
                annualised_pd, 6
            )
            facility_level.loc[index, "1-year PD"] = np.round(
                1 - survival[min(11, len(pd_df) - 1)], 6
            )
            facility_level.loc[index, "Lifetime PD"] = np.round(
                lifetime_pd, 6
            )
            facility_level.loc[index, "Average LGD"] = np.round(
                lifetime_lgd, 4
            )
            facility_level.loc[index, "Average LGD Adjusted"] = np.round(
                lifetime_lgd_adj, 4
            )
            facility_level.loc[index, "Loss"] = np.round(
                lifetime_loss[-1], 2
            )
            facility_level.loc[index, "Loss Adjusted"] = np.round(
                lifetime_loss_adj[-1], 2
            )
            facility_level.loc[index, "Total Exposure"] = np.round(
                ead_total, 2
            )
            facility_level.loc[index, "Loss0"] = np.round(loss0, 2)
            facility_level.loc[
                index, "LGD Correlation"
            ] = lgd_correlations[region + "_" + pdmodel]

            df2.loc[
                df2["AccountID"] == account_ID, "weighted_PD"
            ] = np.round(annualised_pd, 6)
            df2.loc[
                df2["AccountID"] == account_ID, "weighted_LGD"
            ] = np.round(lifetime_lgd_adj, 6)

            if len(pd_df) == 1:
                df2.loc[
                    df2["AccountID"] == account_ID, "weighted_Collateral"
                ] = np.round(collateral[0], 2)
            else:
                df2.loc[
                    df2["AccountID"] == account_ID, "weighted_Collateral"
                ] = np.round(
                    collateral[: min(11, len(pd_df) - 1)].mean(), 2
                )

            df2.loc[
                df2["AccountID"] == account_ID, "weighted_ECL"
            ] = np.round(ecl, 2)

            # add other summary variables to facility level data - used to create summary dataset which
            # is used as starting point for projections
            start_date = datetime.strptime(
                facilities_row["OpenDate"], "%d/%m/%Y"
            )
            if (
                    pd.isna(facilities_row["MaturityDate"])
                    or facilities_row["MaturityDate"] is None or facilities_row["MaturityDate"] == ""
            ):
                mat_date = pd.NA
                term = pd.NA
            else:
                mat_date = datetime.strptime(
                    facilities_row["MaturityDate"], "%d/%m/%Y"
                )
                term = diff_month(mat_date, start_date)

            age = diff_month(current_date, start_date)

            facility_level.loc[index, "Age"] = age
            facility_level.loc[index, "Term"] = term

            if age <= 12:
                facility_level.loc[index, "Backbook"] = 0
            else:
                facility_level.loc[index, "Backbook"] = 1
        except Exception as e:
            raise Exception(
                "Error summarising " + str(account_ID) + "(" + str(e) + ")"
            )

    return facility_level, df, probationary_file, df2


class Main(trac.TracModel):

    # Set the model parameters
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:
        return trac.declare_parameters(
            trac.P("reporting_date", trac.DATE, label="Reporting date"),
            trac.P("stage_2_probationary_period", trac.INTEGER, label="Stage 2 probationary period (months)", default_value=12),
            trac.P("account_id_column", trac.STRING, label="Account ID column", default_value="AccountID"),
            trac.P("lgd_macro_engine_off", trac.BOOLEAN, label="LGD macro engine off"),
            trac.P("pd_curve", trac.STRING, label="PD curve type to use", default_value="PiT PD"),
            trac.P("use_curve", trac.INTEGER, label="PD curve ID to use", default_value=3)
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        facilities_data_schema = trac.load_schema(schemas, "facilities_data_schema.csv")
        staging_data_schema = trac.load_schema(schemas, "staging_data_schema.csv")
        stage_2_definition_schema = trac.load_schema(schemas, "stage_2_definition_schema.csv")
        stage_3_definition_schema = trac.load_schema(schemas, "stage_3_definition_schema.csv")
        retail_ccf_schema = trac.load_schema(schemas, "retail_ccf_schema.csv")
        generic_transition_matrix_schema = trac.load_schema(schemas, "non_retail_rating_master_scale_schema.csv")
        retail_rating_master_scale_schema = trac.load_schema(schemas, "retail_rating_master_scale_schema.csv")
        staging_external_rating_mapping_schema = trac.load_schema(schemas, "staging_external_rating_mapping_schema.csv")
        scenario_definitions_schema = trac.load_schema(schemas, "scenario_definitions_schema.csv")
        gcc_corporate_pd_term_structures_schema = trac.load_schema(schemas, "gcc_corporate_pd_term_structures_schema.csv")
        gcc_retail_pd_term_structures_schema = trac.load_schema(schemas, "gcc_retail_pd_term_structures_schema.csv")
        uk_corporate_pd_term_structures_schema = trac.load_schema(schemas, "uk_corporate_pd_term_structures_schema.csv")
        ead_prepayment_curves_schema = trac.load_schema(schemas, "ead_prepayment_curves_schema.csv")
        return {
            "facilities_data": trac.ModelInputSchema(facilities_data_schema),
            "staging_data": trac.ModelInputSchema(staging_data_schema),
            "stage_2_definition": trac.ModelInputSchema(stage_2_definition_schema),
            "stage_3_definition": trac.ModelInputSchema(stage_3_definition_schema),
            "retail_ccf": trac.ModelInputSchema(retail_ccf_schema),
            "corporate_rating_master_scale": trac.ModelInputSchema(generic_transition_matrix_schema),
            "retail_rating_master_scale": trac.ModelInputSchema(retail_rating_master_scale_schema),
            "hnwi_rating_master_scale": trac.ModelInputSchema(generic_transition_matrix_schema),
            "sme_rating_master_scale": trac.ModelInputSchema(generic_transition_matrix_schema),
            "staging_external_rating_mapping": trac.ModelInputSchema(staging_external_rating_mapping_schema),
            "uk_scenario_definitions": trac.ModelInputSchema(scenario_definitions_schema),
            "gcc_scenario_definitions": trac.ModelInputSchema(scenario_definitions_schema),
            "gcc_corporate_pd_term_structures": trac.ModelInputSchema(gcc_corporate_pd_term_structures_schema),
            "gcc_retail_pd_term_structures": trac.ModelInputSchema(gcc_retail_pd_term_structures_schema),
            "uk_corporate_pd_term_structures": trac.ModelInputSchema(uk_corporate_pd_term_structures_schema),
            "uk_sme_pd_term_structures": trac.ModelInputSchema(uk_corporate_pd_term_structures_schema),
            "uk_hnwi_pd_term_structures": trac.ModelInputSchema(uk_corporate_pd_term_structures_schema),
            "ead_prepayment_curves": trac.ModelInputSchema(ead_prepayment_curves_schema),
        }

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        ecl_schema = trac.load_schema(schemas, "ecl_calculation_schema.csv")

        return {
            "ecl_calculation": trac.define_output_table(ecl_schema.table.fields, label="ECL calculation"),
        }

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")
        account_id_column = ctx.get_parameter("account_id_column")
        stage_2_probationary_period = ctx.get_parameter("stage_2_probationary_period")
        use_curve = ctx.get_parameter("use_curve")
        lgd_macro_engine_off = ctx.get_parameter("lgd_macro_engine_off")

        # The individual exposures that we are calculating the staging of
        facilities_data = ctx.get_pandas_table("facilities_data")
        facilities_data = facilities_data.head(100)
        staging_data = ctx.get_pandas_table("staging_data")

        # The grades and notches of the external ratings
        stage_2_definition = ctx.get_pandas_table("stage_2_definition")
        stage_3_definition = ctx.get_pandas_table("stage_3_definition")

        retail_ccf = ctx.get_pandas_table("retail_ccf")

        corporate_rating_master_scale = ctx.get_pandas_table("corporate_rating_master_scale")
        retail_rating_master_scale = ctx.get_pandas_table("retail_rating_master_scale")
        hnwi_rating_master_scale = ctx.get_pandas_table("hnwi_rating_master_scale")
        sme_rating_master_scale = ctx.get_pandas_table("sme_rating_master_scale")

        staging_external_rating_mapping = ctx.get_pandas_table("staging_external_rating_mapping")

        retail_ccf.rename(columns={"Behavioural_Term_Months": "Behavioural Term (Months)"}, inplace=True)
        facilities_data['n'] = arange(len(facilities_data)) + 1
        staging_external_rating_mapping = staging_external_rating_mapping.to_numpy()

        # The names of the scenarios, their type and their likelihood
        gcc_scenario_definitions = ctx.get_pandas_table("gcc_scenario_definitions")
        uk_scenario_definitions = ctx.get_pandas_table("uk_scenario_definitions")

        gcc_scenario_definitions["Use"] = True
        uk_scenario_definitions["Use"] = True

        gcc_corporate_pd_term_structures = ctx.get_pandas_table("gcc_corporate_pd_term_structures")
        gcc_retail_pd_term_structures = ctx.get_pandas_table("gcc_retail_pd_term_structures")
        uk_corporate_pd_term_structures = ctx.get_pandas_table("uk_corporate_pd_term_structures")
        uk_sme_pd_term_structures = ctx.get_pandas_table("uk_sme_pd_term_structures")
        uk_hnwi_pd_term_structures = ctx.get_pandas_table("uk_hnwi_pd_term_structures")

        ead_prepayment_curves = ctx.get_pandas_table("ead_prepayment_curves")
        ead_prepayment_curves.rename(columns={"Months_on_Book": "Months on Book"}, inplace=True)

        lgd_curve = "TTC" if lgd_macro_engine_off else "PiT"

        # mrs is dictionary of the four datasets, then gets called internal_rating_matrix
        st2 = stage_2_definition
        stage_2_dict = dict(zip(list(st2.Name), list(st2.Value)))

        st3 = stage_3_definition
        stage_3_dict = dict(zip(list(st3.Name), list(st3.Value)))

        logger.info(f"Generating the ECL for a {reporting_date} reporting date")

        # Keep only the staging data for those have a facility entry
        ids = facilities_data[account_id_column].tolist()

        # TODO is this actually probationary data that says who is on probation
        staging_datay = staging_data.loc[staging_data[account_id_column].isin(ids)]

        mat = {
            "Corporate": corporate_rating_master_scale,
            "HNWI": hnwi_rating_master_scale,
            "SME": sme_rating_master_scale,
            "Retail": retail_rating_master_scale
        }

        scenarios = {
            "UK": uk_scenario_definitions,
            "GCC": gcc_scenario_definitions
        }

        pd_curves = {
            "UK_Corporate": uk_corporate_pd_term_structures,
            "UK_SME": uk_sme_pd_term_structures,
            "UK_HNWI": uk_hnwi_pd_term_structures,
            "GCC_Corporate": gcc_corporate_pd_term_structures,
            "GCC_Retail": gcc_retail_pd_term_structures,
        }

        lgd_correlations = {
            'GCC_Corporate': 0.12,
            'GCC_Retail': 0.05,
            'UK_Corporate': 0.12,
            'UK_HNWI': 0.12,
            'UK_SME': 0.05
        }

        no_collateral_recovery_rates = {
            "sov": 0.59,
            "non_sov_loan": 0.59,
            "non_sov_bond": 0.469,
        }

        cost_of_recovery = {
            "property": 0.05,
            "other_collateral": 0.05,
            "no_collateral": None,
        }

        lgd_floor = {
            "property": 0.01,
            "other_collateral": 0.01,
            "no_collateral": None,
        }

        time_to_recovery = {
            "property": {
                "Base":   26,
                "Upside":   26,
                "Downside":  30
        },
            "other_collateral": 14.4,
            "no_collateral": 14.4,
        }

        haircuts = {
            "other_collateral": {
                "Invincible": 0,
                "Strong": 0.1,
                "Good": 0.25,
                "Satisfactory": 0.4,
                "Poor": 0.6,
                "Very poor": 0.8,
            },
            "property": {
                "House": (0.2, 0.3),
                "Flat, Block": (0.25, 0.35),
                "Student Accommodation": (0, 0.35),
                "Hotel, Serviced Flat, Mixed, Invested in Real Estate": (0, 0.4),
                "Office": (0, 0.45)
            }
        }

        ppd_parameters = {0.0: 0.15, 0.45: 0.2, 0.6: 0.35, 0.75: 0.5, 2.0: 1.0}

        # TODO - don't know what this is the projection of
        projections = [0.02595984978868266, 0.03044328732204437, 0.03492672485540609, 0.039410162388767815, 0.039410162388767815, 0.039410162388767815, 0.039410162388767815, 0.039410162388767815, 0.039410162388767815, 0.039410162388767815, 0.03953638690177733, 0.03966261141710143, 0.03978883593011095, 0.039915060443120466, 0.040041284956129984, 0.040167509469139516, 0.040293733982149034, 0.04041995849747312, 0.04054618301048265, 0.04054618301048265, 0.04054618301048265, 0.04054618301048265, 0.04067240752349217, 0.04079863203881625, 0.04092485655182577, 0.041051081064835286, 0.04117730558015938, 0.0413035300931689, 0.0413035300931689, 0.0413035300931689, 0.0413035300931689, 0.04142975460617842, 0.041555979121502516, 0.041682203634512034, 0.04180842814752155, 0.04193465266284565, 0.04206087717585517, 0.04206087717585517, 0.04206087717585517, 0.04206087717585517, 0.042187101688864685, 0.042313326204188796, 0.042439550717198314, 0.042439550717198314, 0.042439550717198314, 0.042439550717198314, 0.04256577523020783, 0.04269199974553193, 0.04281822425854145, 0.042944448771550965, 0.04307067328687506, 0.04319689779988458, 0.04319689779988458, 0.04319689779988458, 0.04319689779988458, 0.0433231223128941, 0.043449346828218195, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771, 0.04357557134122771]

        ecl, cashflows, staging, scenarios = CalculateECL(
            reporting_date=reporting_date,
            facilities=facilities_data,
            ccf_df=retail_ccf,
            mat=mat,
            stage_3_dict=stage_3_dict,
            probationary_file=staging_datay,
            stage_2_probationary_period=stage_2_probationary_period,
            staging_external_rating_mapping=staging_external_rating_mapping,
            scenarios=scenarios,
            pd_curves=pd_curves,
            stage_2_dict=stage_2_dict,
            ead_prepayment_curves=ead_prepayment_curves,
            revolving_function=cal_EAD_term_structure_revolving,
            amortisation_function=cal_EAD_term_structure_amortising_3m_interest,
            use_curve=use_curve,
            lgd_curve=lgd_curve,
            lgd_correlations=lgd_correlations,
            no_collateral_recovery_rates=no_collateral_recovery_rates,
            cost_of_recovery_parameters=cost_of_recovery,
            lgd_floor_parameters=lgd_floor,
            time_to_recovery_parameters=time_to_recovery,
            haircuts=haircuts,
            ppd_parameters=ppd_parameters,
            projections=projections
        )

        ecl["OpenDate"] = pd.to_datetime(ecl["OpenDate"], format='%d/%m/%Y')
        ecl["MaturityDate"] = pd.to_datetime(ecl["MaturityDate"], format='%d/%m/%Y', errors="coerce")

        ecl.rename(
            columns={
                "1-year PD": "ONE_YEAR_PD",
                "Lifetime PD": "LIFETIME_PD",
                "Average LGD": "AVERAGE_LGD",
                "Average LGD Adjusted": "AVERAGE_LGD_ADJUSTED",
                "Loss Adjusted": "LOSS_ADJUSTED",
                "Total Exposure": "TOTAL_EXPOSURE",
                "Behavioural Term": "BEHAVIOURAL_TERM",
                "Annualised 1-year PD": "ANNUALISED_ONE_YEAR_PD",
                "LGD 0": "LGD_0",
                "LGD Correlation": "LGD_CORRELATION",
            }, inplace=True
        )

        ecl["Stage"] = ecl["Stage"].astype(int)
        ecl["Age"] = ecl["Age"].astype(int)
        ecl["BEHAVIOURAL_TERM"] = ecl["BEHAVIOURAL_TERM"].astype(int)
        # Output the datasets
        ctx.put_pandas_table("ecl_calculation", ecl)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "trac_poc/config/calculate_ecl.yaml", "trac_poc/config/sys_config.yaml")
