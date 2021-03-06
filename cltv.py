########## Cltv ##########

# Variables
# InvoiceNo: Invoice number. The unique number of each transaction, namely the invoice. Delete operation if it starts with C.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.

################ Data Understanding ################

##### Importing Libraries#############

import datetime as dt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##### Import Data ########

# We will use 2010-2011 sheet in the Online Retail II excel ########
df_ = pd.read_excel("/Users/ilaydakursun/Desktop/Bootcamp/Hafta 3/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()  # Copy of DataFrame

df.isnull().any()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]
## Price must be greater than 0 #####
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]


##### Data Preparation #######

df_uk= df[df['Country'] == 'United Kingdom']
df_uk.info()


##### Descriptive Statistics#######

df_uk.describe([0.01, 0.05, 0.1, 0.90, 0.95, 0.99]).T


## We need Total Price column######
df_uk['TotalPrice'] = df_uk['Quantity'] * df_uk['Price']

today_date = dt.datetime(2011, 12, 11)


################ Customer Life Time Value Predic ################
#  Creation of metrics to be used for the cltv calculate #
cltv_df = df_uk.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary'] # T is age of clients

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df["recency"] = cltv_df["recency"] / 7 .
cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["frequency"].astype(int)
########################## BG-NBD MODEL ##########################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])


########################## GAMMA-GAMMA MODEL ##########################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],cltv_df['monetary'])


###### Task 1: 1 ay ##############
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1, #1 ay
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv= cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# 6 AYLIK

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6, #6 ay
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# G??REV 2 : 2010-2011 UK m????terileri i??in 1 ayl??k ve 12 ayl??k CLTV hesaplay??n??z.
# TASK 2
# Farkl?? zaman periyotlar??ndan olu??an CLTV analizi
# 1 ayl??k CLTV'de en y??ksek olan 10 ki??i ile 12 ayl??k'taki en y??ksek 10 ki??iyi analiz ediniz.
# Fark var m??? Varsa sizce neden olabilir?


# S??f??rdan model kurulmas??na gerek yoktur.
# ??nceki soruda olu??turulan model ??zerinden ilerlenebilir.
#12 ayl??k
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(10)


cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# 12 ay'??n g??zlem de??erlerinde CLTV ??OK Y??KSEK

# 1 ayl??k cltv en y??ksek 10 ki??i  analiz et

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])



cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10)
# 1 ayl??k g??zlem de frequency de??erinin etkisinin ??ok b??y??k oldu??unu g??r??l??yor


# 12 ayl??k analiz et
# 1 ayl????a k??yaslarsak yakla????k 12 kat artt??????n?? g??r??l??yor

bgf.predict(4*12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])



cltv_df.sort_values("expected_purc_12_month", ascending=False).head(10)


# G??REV 3: Segmentlerin olu??turulmas??

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_cltv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_cltv", ascending=False).head(50)

cltv_final["segment"] = pd.qcut(cltv_final["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.sort_values(by="clv", ascending=False).head(10)


cltv_final.groupby("segment").agg({"count", "mean", "sum"})
