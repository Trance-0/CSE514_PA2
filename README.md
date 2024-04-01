# CSE514_PA2
CSE 514 Programming assignment

## Task assignment

Dijkstra Liu:

- Artificial Neural Network
- Supported Vector Machine

Zheyuan Wu:

- K-Nearest Neighbors
- Random Forest

## Dataset

[Customer personality analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

## Notes

### import dataset

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
