import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv(r'C:/Users/navee/Documents/Python Project/Expanded_Sample_Student_Task_Dataset.csv')


data['Task_Difficulty'] = data['Task_Difficulty'].astype(str)
data['Task_Type'] = data['Task_Type'].astype(str)
data['Break_Preference'] = data['Break_Preference'].astype(str)
data['High_Screen_Time'] = (data['Screen_Time'] > 2).astype(str)
data['High_Stress_Level'] = (data['Stress_Level'] > 7).astype(str)
data['Low_Grades'] = (data['Grades'] < 60).astype(str)

transactions = data[['Task_Difficulty', 'Task_Type', 'Break_Preference', 
                     'High_Screen_Time', 'High_Stress_Level', 'Low_Grades']].values.tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))

rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(rules)
