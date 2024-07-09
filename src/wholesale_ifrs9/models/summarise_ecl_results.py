import datetime
import typing as tp
from typing import List, Any, Union

import numpy as np
import pandas as pd
# Load the TRAC runtime library
import tracdap.rt.api as trac

# Load the schemas library
from trac_poc import schemas as schemas

"""
A model that performs business logic validations on the facility data.
"""


def wmean_grouped2(group, var_name_in, var_name_weight):
    d = group[var_name_in]
    w = group[var_name_weight]
    return np.round((d * w).sum() / w.sum(), 2)


FUNCS = {"mean": np.mean, "sum": np.sum, "count": np.count_nonzero}


def my_summary2(
        data,
        var_names_in,
        var_names_out,
        var_functions,
        var_name_weight=None,
        var_names_group=None,
):
    result = pd.DataFrame()

    if var_names_group is None:
        grouped = data.groupby(lambda x: True)
    else:
        grouped = data.groupby(var_names_group)

    for var_name_in, var_name_out, var_function in zip(var_names_in, var_names_out, var_functions):
        if var_function == "wmean":
            func = lambda x: wmean_grouped2(x, var_name_in, var_name_weight)
            result[var_name_out] = pd.Series(grouped.apply(func))
        elif var_function == "count":
            func = FUNCS[var_function]
            result[var_name_out] = grouped[var_name_in].apply(func)
        else:
            # This originally had a round
            func = FUNCS[var_function]
            result[var_name_out] = grouped[var_name_in].apply(func)

    result = result.reset_index()

    return result


class Main(trac.TracModel):

    # Set the model parameters
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:
        return trac.declare_parameters(
            trac.P("reporting_date", trac.DATE, label="Reporting date")
        )

    # Set the model input datasets
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        ecl_calculation_schema = trac.load_schema(schemas, "ecl_calculation_schema.csv")

        return {"ecl_calculation": trac.ModelInputSchema(ecl_calculation_schema)}

    # Set the model output datasets
    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        ecl_summary_schema = trac.load_schema(schemas, "ecl_summary_schema.csv")

        return {"ecl_summary": trac.define_output_table(ecl_summary_schema.table.fields, label="ECL calculation summary")}

    def run_model(self, ctx: trac.TracContext):
        # Set up the logger
        logger = ctx.log()

        # Load the input data
        ecl_calculation = ctx.get_pandas_table("ecl_calculation")

        # Load the parameters
        reporting_date = ctx.get_parameter("reporting_date")

        logger.info(f"Creating ECL calculation summary")

        group_columns = [
            "Region",
            "PDModel",
            "Segment",
            "Stage",
            "Backbook",
            "PaymentType",
            "RateType",
            "CollateralType",
        ]

        summary = my_summary2(
            data=ecl_calculation,
            var_names_in=[
                "Exposure",
                "Exposure",
                "ANNUALISED_ONE_YEAR_PD",
                "ONE_YEAR_PD",
                "LIFETIME_PD",
                "TTC_PD",
                "LGD_0",
                "LGD_0",
                "ECL",
                "InterestRate",
                "RWA",
                "Loss",
                "Loss0",
                "LOSS_ADJUSTED",
                "TOTAL_EXPOSURE",
                "BEHAVIOURAL_TERM",
                "LGD_CORRELATION",
                "Age",
                "Term",
            ],
            var_names_out=[
                "Volume",
                "Exposure",
                "AVG_ANNUALISED_ONE_YEAR_PD",
                "AVG_ONE_YEAR_PD",
                "AVG_LIFETIME_PD",
                "AVG_TTC_PD",
                "AVG_LGD_T0",
                "AVG_EXPOSURE_WEIGHTED_LGD_T0",
                "SUM_ECL",
                "AVG_EXPOSURE_WEIGHTED_INTEREST_RATE",
                "SUM_RWA",
                "SUM_Loss",
                "SUM_Loss0",
                "SUM_LOSS_ADJUSTED",
                "SUM_TOTAL_EXPOSURE",
                "AVG_BEHAVIOURAL_TERM",
                "WEIGHTED_AVG_LGD_CORRELATION",
                "AVG_Age",
                "AVG_Term",
            ],
            var_functions=[
                "count",
                "sum",
                "mean",
                "mean",
                "mean",
                "mean",
                "mean",
                "wmean",
                "sum",
                "wmean",
                "sum",
                "sum",
                "sum",
                "sum",
                "sum",
                "mean",
                "wmean",
                "mean",
                "mean",
            ],
            var_name_weight="Exposure",
            var_names_group=group_columns,
        )

        summary["Date"] = reporting_date

        # Output the dataset
        ctx.put_pandas_table("ecl_summary", summary)


if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_model(Main, "trac_poc/config/summarise_ecl_results.yaml", "trac_poc/config/sys_config.yaml")
