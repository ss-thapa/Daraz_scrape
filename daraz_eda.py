import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.api as sm 
from scipy.stats import probplot




pd.set_option('display.max_column', None)
pd.set_option('display.max_colwidth', None)




df = pd.read_csv('/Users/sunilthapa/Desktop/programming/datahub/darazdata/daraz_data_all_pages.csv')


df['product_name'] = df['product_name'].str.strip()

df= df.drop_duplicates()

df['ratings'] = df['ratings'].str.replace('/5', ' ')


df['ratings'] = df['ratings'].str.replace('-May', ' ')


df['ratings'] = df['ratings'].astype(float)

df['no_of_ratings'] = df['no_of_ratings'].astype(str).str.replace('-', '')

df['no_of_ratings'] = df['no_of_ratings'].astype(int)

df['sold_num'] =df['sold_num'].str.replace(' Sold', '')

def convert_k_to_1000(value):
    if isinstance(value, str) and 'k' in value.lower():
        try:
            return int(float(value[:-1]) * 1000)
        except ValueError:
            return 0  # Handle the case where 'k' is not followed by a numeric value
    else:
        return int(value)

# Apply the function to the 'no_of_ratings' column
df['sold_num'] = df['sold_num'].apply(convert_k_to_1000)

df['original_price'] = df['original_price'].replace('Rs. ', '', regex=True).replace(',', '', regex=True).astype(int)

df['current_price'] = df['current_price'].replace('Rs.', '', regex=True).replace(',', '', regex=True).astype(int)

df = df[df['current_price'] > 50]

df['discount_amount'] = df['original_price']-df['current_price']


df['discount_amount'] = np.maximum(df['discount_amount'], 0)


### deleting 3 datas 
index_to_delete = [3472, 1632, 519]  # Specify the index of the row you want to delete

# Use drop to remove the row by index
df = df.drop(index_to_delete)




### feature enginnering on product_name column about price

df['contains_keywords'] = df['product_name'].str.contains(r'\b(combo|pack|pcs)\b', case=False, regex=True)

df['contains_numeric'] = df['product_name'].str.contains(r'\d', regex=True)

df['extracted_numeric'] = df[df['contains_keywords'] & df['contains_numeric']]['product_name'].str.extract(r'(\b\d\b)')

pattern = r'\b(?:Five|Six|Seven|One|Two|Three|Four|Eight|Nine|Ten)\b'

# Create a new column indicating whether the product_name contains any of the specified words
df['contains_numeric_words'] = df['product_name'].str.contains(pattern, case=False, regex=True)


df =df[~(df['contains_numeric_words'] & df['contains_keywords'])]

df = df.drop(columns=['contains_keywords', 'contains_numeric', 'contains_numeric_words'])



df['extracted_numeric'] = df['extracted_numeric'].fillna(0).astype(int)



df['price_of_1'] = np.where(df['extracted_numeric'] != 0, df['current_price'] / df['extracted_numeric'], df['current_price'])


df = df.drop(columns=['extracted_numeric'])


### creating brand name category


brand_mapping = {'shangri': 'Shangrila','police':'police', 'being human':'being human', 'doro':'Doro', 'nyptra':'nyptra', 'hummel':'hummel',
                'hills and clouds':'Hills And Clouds', 'kilometer':'kilometer', 'bossini':'Bossini', 'piazzaitalia':'piazzaitalia','zamz':'zamz','livingtex':'livingtex',
                'trikaya':'trikaya','oxemberg':'oxemberg', 'creative touch':'creative touch','dockers':'dockers', 'anta':'Anta', 'logo':'logo', 'fuloo':'Fuloos',
                'j.fisher':'J.Fisher','Gents park':'Gents Park', 'wrogn':'wrogn', 'pepe jeans':'pepe jeans', 'Van heusen':'Van Heusen', 'Benetton':'Benetton', 'Bumchums':'Bumchums',
                'Gymwolf':'Gymwolf','Binay Embroidery':'Binay Embroidery'}

df['brand_name'] = 'Other'

# Loop through the dictionary and update 'brand_name' based on the word found in 'product_name'
for word, brand_name in brand_mapping.items():
    df.loc[df['product_name'].str.contains(word, case=False), 'brand_name'] = brand_name

# If 'brand_name' is still 'Other', it means no match was found
df.loc[df['brand_name'] == 'Other', 'brand_name'] = 'Other'




df['ratings'] = df['ratings'].replace(0, np.NAN)

df['price_of_1'] = df['price_of_1'].replace(0, np.NAN)

df['no_of_ratings'] = df['no_of_ratings'].replace(0, np.NAN)











df1 = df[df['brand_name'] == 'Shangrila']


# sns.kdeplot(df1['price_of_1'])
# plt.show()

### plotting QQ plot to check is the price of 1 column is normal distribution or not using pure math

# tempdf = sorted(df1['price_of_1'].tolist())

# y_quant = []

# for i in range(1, 101):
#     y_quant.append(np.percentile(tempdf,i))

# sample_normal_dis_data = np.random.normal(loc=0,scale=1,size=1000)

# x_quant = []

# for i in range(1,101):
#     x_quant.append(np.percentile(sample_normal_dis_data, i))

# # sns.scatterplot(x=x_quant,y=y_quant)
# # plt.show()


# ## using stats model plotting QQ plot

# fig = sm.qqplot(df1['price_of_1'], line='45', fit=True)
# plt.show()






## Converting price_of_1 to normal distribution assuming it is log normal to check if the price of 1 is log normal distribution or not 

# data = np.array(df1['price_of_1'])

# # Calculate the logarithm (base 10) of each data point
# log_data = np.log10(data)


# tempdf = sorted(log_data.tolist())

# y_quant = []

# for i in range(1, 101):
#     y_quant.append(np.percentile(tempdf,i))

# sample_normal_dis_data = np.random.normal(loc=0,scale=1,size=1000)

# x_quant = []

# for i in range(1,101):
#     x_quant.append(np.percentile(sample_normal_dis_data, i))

# sns.scatterplot(x=x_quant,y=y_quant)
# plt.show()




# plt.figure(figsize=(10, 6))
# sns.histplot(df1['price_of_1'], bins=30, kde=True)
# plt.title('Histogram of price_of_1')
# plt.show()

# Q-Q plot
plt.figure(figsize=(10, 6))
probplot(df1['price_of_1'], dist='norm', plot=plt)
plt.title('Q-Q plot of price_of_1')




fig = sm.qqplot(df1['price_of_1'], line='45', fit=True)
plt.show()
