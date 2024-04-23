"""return data safe to process

call function init to get raw, well-defined complete dataset.

the code using this module to import the data should have the following structure

```filetree
- project_folder
    - data
        - __init__.py
        - customer_personality_analysis.py
        - marketing_campaign.csv
    - part_A
        - your_code.py
```

and in `your_code.py`, import the module using the example code follow:

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

    def numerical_encode(self,df=None,bisected_data=False):
        """ returns numerical encoded categorical value for customer

        Args:
            df: dataframe with the same column as defined in this class, self.df by default
            bisected_data: whether the data is already bisected by income level and split to different populations.

        Returns:
            "data": numerical encoded self.df as DataFrame, to preserve the column names
            "numerical": self.numerical,
            "categorical": self.categorical,
        """
        encoded_df=self.df.copy()
        if (df is not None):
            encoded_df=df.copy()
        print(encoded_df.head())
        # Convert education by cycle of education, ordinal encoding, Master and under grads a
        encoded_df['Education'] = encoded_df['Education'].replace({'Basic': 1, '2n Cycle': 2, 'Graduation': 2,'PhD': 3, 'Master': 2})
        # Convert Dt_customer by day of join
        encoded_df['Dt_Customer']=pd.to_datetime(encoded_df['Dt_Customer'], format='%d-%m-%Y')
        encoded_df['Dt_Customer'] = pd.to_timedelta(encoded_df['Dt_Customer'].max()-encoded_df['Dt_Customer']).dt.total_seconds().astype(int)
        # one-hot encoding for marital status and income level (if exists)
        # If data is bisected, we don't need marital status column anymore.
        if not bisected_data:
            # get the dummies and store it in a variable
            dummies = pd.get_dummies(encoded_df['Marital_Status'])
            # Concatenate the dummies to original dataframe
            encoded_df= pd.concat([encoded_df, dummies], axis=1)
        # encode income level
        else:
            encoded_df['Income_level'].replace(['hi', 'low'],
                        [1,0], inplace=True)
            # encoded_df=encoded_df.drop(['Income_level'])
        encoded_df=encoded_df.drop(['Marital_Status'],axis=1)
        return {
            "data": encoded_df,
            "numerical": self.numerical_split_population if bisected_data else self.numerical,
            "categorical": self.categorical_split_population if bisected_data else self.numerical,
        }

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

    def prepared_standardize_data(self, data_name="married", debug=False):

        data = []
        if data_name == "married":
            data = self.married_data()['data']
        elif data_name == "partner_loss":
            data = self.partner_loss_data()['data']
        else:
            data = self.single_data()['data']

        cpa_data = self.numerical_encode(data, bisected_data=True)['data']

        X = cpa_data.iloc[:, :-1].values
        y = cpa_data.iloc[:, -1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if debug:
            print(X, y)

        return X, y

    def prepare_reduced_data(self, data_name="married"):

        cpa = customer_personality_analysis()
        X, y = cpa.prepared_standardize_data(data_name=data_name, debug=False)


        pca = PCA(n_components=9)
        X_pca = pca.fit_transform(X)

        return X_pca, y

