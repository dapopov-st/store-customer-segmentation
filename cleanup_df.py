import pandas as pd
data = pd.read_csv('./data/final_clustered.csv',index_col=0)
df=data[['ID','KM3','GM3','MntTotal', 'MntSpentNorm',
         'Income','Len_Customer', 'HasChildren',
          'HasPartner', 'NumChildren','age','AcceptedCmp1',
          'AcceptedCmp2','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5','AcceptedCmpTot', 'Response', 'Complain',
          'Year_Birth', 'Education', 'Marital_Status', 'Kidhome',
          'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
          'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
          'MntGoldProds', 'NumTotalPurchases','NumDealsPurchases', 'NumWebPurchases',
          'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth','Response', 'Complain','KM2', 'AC3', 'AC2', 'GM2','LogIncome',]]


df['MntWinesNorm']=df['MntWines']/df['Len_Customer']

df['MntFruitsNorm']=df['MntFruits']/df['Len_Customer']

df['MntMeatProductsNorm']=df['MntMeatProducts']/df['Len_Customer']

df['MntFishProductsNorm']=df['MntFishProducts']/df['Len_Customer']

df['MntSweetProductsNorm']=df['MntSweetProducts']/df['Len_Customer']

df['MntGoldProdsNorm']=df['MntGoldProds']/df['Len_Customer']

df['NumDealsPurchasesNorm']=df['NumDealsPurchases']/df['Len_Customer']

df['NumWebPurchasesNorm']=df['NumWebPurchases']/df['Len_Customer']

df['NumStorePurchasesNorm']=df['NumStorePurchases']/df['Len_Customer']

df['NumWebVisitsMonthNorm']=df['NumWebVisitsMonth']/df['Len_Customer']

df['NumCatalogPurchasesNorm']=df['NumCatalogPurchases']/df['Len_Customer']
df['SpendPropOfTotal'] = (df['MntTotal']/df['Len_Customer'])/(df['Income']/796) #Daily spending/Daily Income
df['AvgPerPurchase'] = df['MntTotal']/df['NumTotalPurchases']
df['NumTotalPurchasesNorm']=df['NumTotalPurchases']/df['Len_Customer']

df.to_csv('data/final_clustered_extra_cols.csv')