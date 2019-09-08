import luigi
import numpy as np
import pandas as pd
from fpdf import FPDF

class WeekOnePartOne(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(path='../reports/week_one_part_one.pdf')

    def run(self):
        intervals = np.linspace(0, 1)
        x = np.cos(intervals)
        y = np.sin(intervals)
        combined = np.concatenate((x, y))

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(800, 5, str(combined))
        pdf.output(self.output().path, 'F')

class WeekOnePartTwo(luigi.Task):
    def requires(self):
        return [WeekOnePartOne()]

    def output(self):
        return luigi.LocalTarget('../reports/week_one_part_two.pdf')

    def run(self):
        ran_array = np.random.randn(3,5)
        sum_all_val = np.sum(ran_array)
        sum_column = np.sum(ran_array, axis=0)
        sum_row = np.sum(ran_array, axis=1)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(400, 5, 'Sum all every value:' + str(sum_all_val) + '\n' +
                       'Sum of the column values:' + str(sum_column) + '\n' +
                       'Sum of the row values:' + str(sum_row)
                       )
        pdf.output(self.output().path, 'F')

class WeekOnePartThree(luigi.Task):
    def requires(self):
        return [WeekOnePartTwo()]

    def output(self):
        return luigi.LocalTarget("../reports/week_one_part_three.pdf")

    def run(self):
        ran_array_2 = np.random.randn(5,5)
        df = pd.DataFrame({'Column1':ran_array_2[:,0],'Column2':ran_array_2[:,1],
                                'Column3':ran_array_2[:,2],'Column4':ran_array_2[:,3],
                                'Column5':ran_array_2[:,4]})
        df.sort_values(by='Column2', inplace=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 10)
        print(df)
        pdf.multi_cell(400, 5, str(df))
        pdf.output(self.output().path, 'F')

if __name__ == '__main__':
    luigi.run()
