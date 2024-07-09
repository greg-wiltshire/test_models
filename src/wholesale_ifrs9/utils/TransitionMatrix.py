import pandas as pd
from wholesale_ifrs9.utils.ITermStructure import ITermStructure
import json


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
