"""return data safe to process

call function init to get raw, well-defined complete dataset.

the code using this module to import the data should have the following structure

- project_folder
    - data
        - __init__.py
        - customer_personality_analysis.py
        - marketing_campaign.csv
    - part_A
        - your_code.py

and in `your_code`.py, import the module using the example code follow:

```python

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)

from customer_personality_analysis import customer_personality_analysis
 
cpa_data = customer_personality_analysis().data()
```
"""

import os
from pathlib import Path
import pandas as pd


class customer_personality_analysis:

    def __init__(self):
        # load local data
        df = pd.read_csv(
            os.path.join(Path(__file__).resolve().parent, "marketing_campaign.csv"),
            sep="\t",
        )
        df = df.dropna()
        numerical_vals = [
            "Year_Birth",
            "Income",
            "Kidhome",
            "Teenhome",
            "Recency",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "Complain",
        ]
        categorical_vals = ["Education", "Marital_Status", "Dt_Customer"]
        # remove unwanted rows
        self.df = pd.concat([df[numerical_vals], df[categorical_vals]], 
                  axis = 1)
        self.numerical = numerical_vals
        self.categorical = categorical_vals
        self.numerical_split_population = [
            "Year_Birth",
            "Kidhome",
            "Teenhome",
            "Recency",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "Complain",
        ]
        self.categorical_split_population = [
            "Income_level",
            "Education",
            "Marital_Status",
            "Dt_Customer",
        ]

    def one_hot_encode(self,bisected_data=False):
        """ returns one-hot encoded categorical value for the 
        
        """

    def data(self):
        """return packed dictionary of data

        Returns:
            "data": self.df,
            "numerical": self.numerical,
            "categorical": self.categorical,

        """
        return {
            "data": self.df,
            "numerical": self.numerical,
            "categorical": self.categorical,
        }

    def __income_bisector(self, row):
        return "hi" if row["Income"] > 50000 else "low"

    def married_data(self):
        """ population 1 for the problem
        married data are defined with self.df["Marital_Status"] is in values: ["Married", "Together"]

        Returns:
            dataset of married population and categorized income
        """
        married_col_name = ["Married", "Together"]
        population=self.df.copy()
        population = population.loc[self.df["Marital_Status"].isin(married_col_name)]
        population["Income_level"] = population.apply(self.__income_bisector, axis=1)
        population=population.drop(['Income'],axis=1)
        return {
            "data": population,
            "numerical": self.numerical_split_population,
            "categorical": self.categorical_split_population,
        }

    def single_data(self):
        """ population 2 for the problem
        single data are defined with self.df["Marital_Status"] is in values: ["YOLO", "Alone", "Single"]

        Returns:
            dataset of singled population and categorized income
        """
        single_col_name = ["YOLO", "Alone", "Single"]
        population=self.df.copy()
        population = population.loc[self.df["Marital_Status"].isin(single_col_name)]
        population["Income_level"] = population.apply(self.__income_bisector, axis=1)
        population=population.drop(['Income'],axis=1)
        return {
            "data": population,
            "numerical": self.numerical_split_population,
            "categorical": self.categorical_split_population,
        }

    def partner_loss_data(self):
        """ population 3 for the problem
        partner_loss data are defined with self.df["Marital_Status"] is in values: ["Divorced", "Widow", "Absurd"]

        Returns:
            dataset of partner_loss population and categorized income
        """
        partner_loss_col_name = ["Divorced", "Widow", "Absurd"]
        population=self.df.copy()
        population = population.loc[self.df["Marital_Status"].isin(partner_loss_col_name)]
        population["Income_level"] = population.apply(self.__income_bisector, axis=1)
        population=population.drop(['Income'],axis=1)
        return {
            "data": population,
            "numerical": self.numerical_split_population,
            "categorical": self.categorical_split_population,
        }
