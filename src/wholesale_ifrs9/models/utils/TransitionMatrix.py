from abc import ABC, abstractmethod
import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil import parser
import numpy as np
import json

class ITermStructure(ABC):
    @abstractmethod
    def __init__(self, years: int = 10):
        self.__years = years
        self.__pd_ts = None
        self.__marginal_pd_ts = None
        self.__relative_pd_ts = None
        self.__interpolated_pd_ts = None
        self.__ratings = None

    @property
    def projection_years(self):
        return self.__years

    @projection_years.setter
    def projection_years(self, value: int):
        self.__years = value

    @property
    def ratings(self):
        return self.__ratings

    @ratings.setter
    def ratings(self, value):
        self.__ratings = value

    @property
    def cumulative_term_structures(self):
        return self.__pd_ts

    @cumulative_term_structures.setter
    def cumulative_term_structures(self, value: pd.DataFrame):
        self.__pd_ts = value

    @property
    def marginal_term_structures(self):
        return self.__marginal_pd_ts

    @marginal_term_structures.setter
    def marginal_term_structures(self, value: pd.DataFrame):
        self.__marginal_pd_ts = value

    @property
    def relative_term_structures(self):
        return self.__relative_pd_ts

    @relative_term_structures.setter
    def relative_term_structures(self, value: pd.DataFrame):
        self.__relative_pd_ts = value

    @property
    def interpolated_term_structures(self):
        return self.__interpolated_pd_ts

    @property
    def reporting_Date(self):
        return self.__date

    @reporting_Date.setter
    def reporting_Date(self, value: str):
        self.__date = parser.parse(value)

    @interpolated_term_structures.setter
    def interpolated_term_structures(self, value: pd.DataFrame):
        self.__interpolated_pd_ts = value

    def get_cumulative_term_structure_by_rating(self, rating: str):
        return self.cumulative_term_structures.loc[rating, :]

    def get_marginal_term_structure_by_rating(self, rating: str):
        return self.marginal_term_structures.loc[rating, :]

    def get_relative_term_structure_by_rating(self, rating: str):
        return self.relative_term_structures.loc[rating, :]

    def get_interpolated_term_structure_by_rating(self, rating: str):
        return self.interpolated_term_structures.loc[rating, :]

    def linear_interpolate_ts(self):
        record = {}
        ts = self.relative_term_structures
        for t in range(0, self.projection_years):

            month = (t * 12)
            record['M' + str(month)] = ts['Year ' + str(t)]

            for m in range(1, 12):
                month = (t*12) + m
                record['M' + str(month)] = (ts['Year ' + str(t+1)] - ts['Year ' + str(t)])/12 + \
                                           record['M' + str(month-1)]

        df = pd.DataFrame.from_dict(record)
        df = df.transpose().drop('Default', axis=1)
        df["Months"] = df.index.str.replace('M', '0').astype(np.int64)

        for index, row in df.iterrows():
            df.loc[index, "Date"] = self.reporting_Date + relativedelta(months=row["Months"])
        df.index = df["Months"]
        df = df.drop('Months', axis=1)
        self.interpolated_term_structures = df

class TransitionMatrix(ITermStructure):
    def __init__(self, matrix: pd.DataFrame, reporting_date: str, years: int = 10):
        super().__init__(years)
        matrix.index = matrix["Rating"]
        self.__matrix = matrix.drop("Rating", axis=1)
        self.__matrix_multiplied = {}
        self.ratings = matrix.index
        self.reporting_Date = reporting_date
        self._multiply_matrix()
        self.linear_interpolate_ts()

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, value: pd.DataFrame):
        self.__matrix = value
        self._multiply_matrix()

    @property
    def matrix_multiplied(self):
        return self.__matrix_multiplied

    @property
    def matrix_multiplied_json(self):
        mat = {}

        for k in self.__matrix_multiplied.keys():
            mat[k] = self.__matrix_multiplied[k].to_json()

        return json.dumps(mat)

    @matrix_multiplied.setter
    def matrix_multiplied(self, value: pd.DataFrame):
        self.__matrix_multiplied = value

    def _multiply_matrix(self):
        self.__matrix_multiplied[0] = self.__matrix
        for i in range(1, self.projection_years+1):
            self.__matrix_multiplied[i] = self.__matrix_multiplied[i - 1].dot(self.__matrix)
        self._pd_term_structures()

    def _pd_term_structures(self):
        df = pd.DataFrame()
        mdf = pd.DataFrame()
        rdf = pd.DataFrame()

        for i in range(0, self.projection_years+1):
            df['Year ' + str(i)] = pd.Series(self.__matrix_multiplied[i]["Default"], self.ratings)

        mdf['Year 0'] = df['Year 0']
        for i in range(1, self.projection_years+1):
            mdf['Year ' + str(i)] = df['Year ' + str(i)] - df['Year ' + str(i - 1)]

        for i in range(0, self.projection_years+1):
            rdf['Year ' + str(i)] = mdf['Year ' + str(i)] / mdf['Year 0']

        self.cumulative_term_structures = df
        self.marginal_term_structures = mdf
        self.relative_term_structures = rdf