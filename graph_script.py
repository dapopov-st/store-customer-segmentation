
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/final_clustered.csv',index_col=0)


def box_plots(data,cluster_lab="KM3",y="Education"):
    plt.figure(figsize=(9, 7))

    sns.boxplot(x=cluster_lab, y=y, data=data)
    sns.despine(offset=10, trim=True)

    plt.title(f'Cluster labels for {y}')


def bar_charts(data,cluster_lab="KM3",y="Education"):
    plt.figure(figsize=(9, 7))
    palette=sns.color_palette("Set2")
    plot = sns.countplot(x=data[cluster_lab],hue=data[y], palette = palette)

    plot.set_title(f"Cluster Profiles Based On {y}")
    plt.legend()
    plt.show()

def cluster_scatter(data,x_lab,y_lab,cluster_lab):
    #print(f'Scatter plots for {y_lab} vs {x_lab}')
    plt.figure(figsize=(9, 7))
    #palette=sns.color_palette("rocket", as_cmap=True)
    #palette = sns.color_palette("cmo.balance", n_colors=64, desat=0.2)
    plot = sns.scatterplot(data = data,x=data[x_lab], y=data[y_lab],hue=data[cluster_lab],alpha=.5)#,palette=palette)

    plot.set_title(f"Cluster Profiles Over {x_lab} and {y_lab}")
    #plt.legend()
    plt.show()


def bar_charts_large(data,cluster_lab="KM3",y="Education",y_lim=60):

    palette=sns.color_palette("Set2")
    sns.set(font_scale = 1.3)

    x,y = cluster_lab, y

    df1 = data.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('Percent').reset_index()

    g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1,palette=palette,height=8,aspect=2)
    #g.figure(figsize=(14,8))

    g.ax.set_ylim(0,y_lim)

    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) #+ '%'
        txt_x = p.get_x() 
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)
        #plt.figure(figsize=(14,8))
        #plt.figure(figsize=(14,8))
    plt.title(f"Cluster Profiles Based On {y}")



def campaign_bar(data,campaign='AcceptedCmpTot',cluster_lab='KM3'):
    plt.figure(figsize=(14, 8))
    #palette=sns.color_palette("rocket", as_cmap=True)
    sns.countplot(x=data[campaign],hue=data[cluster_lab])

    plt.title(f"Cluster Profiles Based On Campaign {campaign[-1]} Acceptance")
    plt.legend(title='Cluster',loc=1)
    #plt.savefig('./figures/num_total_acceptance.png')
def campaign_bar_all(data, cluster_lab='KM3'):
    for campaign in ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmpTot']:
        campaign_bar(data,campaign=campaign, cluster_lab=cluster_lab)



def cluster_for_medium(data,cluster_lab='KM3'):
    mediums = ['NumWebPurchases','NumWebPurchasesNorm','NumCatalogPurchases','NumCatalogPurchasesNorm',
               'NumStorePurchases','NumStorePurchasesNorm','NumWebVisitsMonth','NumWebVisitsMonthNorm',
                'NumTotalPurchases']
    for medium in mediums:
        box_plots(data,cluster_lab=cluster_lab,y=medium)


def cluster_for_type(data,cluster_lab='KM3'):
    prod_types = ["MntWines","MntWinesNorm","MntMeatProducts","MntMeatProductsNorm",
                "MntFishProducts","MntFishProductsNorm","MntSweetProducts","MntSweetProductsNorm",
                "MntGoldProds","MntGoldProdsNorm"]
    for prod_type in prod_types:
        box_plots(data,cluster_lab=cluster_lab,y=prod_type)


def cluster_for_spent_and_len(data,cluster_lab='KM3'):
    print("CLUSTERING RESULTS OVER AMOUNT SPENT AND LENGTH AS CUSTOMER")
    #cluster_for_spent_and_len(cluster_lab=cluster_lab)
    print("Income vs Total Amount Plot")
    cluster_scatter(data,x_lab='Income',y_lab='MntTotal',cluster_lab=cluster_lab)
    print("Income vs Total Normalized Amount Plot")
    cluster_scatter(data=data,x_lab='Income',y_lab='MntSpentNorm',cluster_lab=cluster_lab)
    box_plots(data,cluster_lab=cluster_lab,y="MntTotal")
    box_plots(data,cluster_lab=cluster_lab,y="MntSpentNorm")
    box_plots(data,cluster_lab=cluster_lab,y="Len_Customer")
    box_plots(data,cluster_lab=cluster_lab,y="SpendPropOfTotal")
    box_plots(data,cluster_lab=cluster_lab,y="AvgPerPurchase")
    box_plots(data,cluster_lab=cluster_lab,y="NumTotalPurchasesNorm")





def cluster_for_demographic(data,cluster_lab='KM3'):
    box_plots(data,cluster_lab=cluster_lab,y="Income")
    box_plots(data,cluster_lab=cluster_lab,y="age")
    bar_charts_large(data,cluster_lab,y='Education',y_lim=60)
    bar_charts_large(data,cluster_lab,y='HasPartner',y_lim=70)
    bar_charts_large(data,cluster_lab,y='NumChildren',y_lim=80)


def cluster_for_recency(data,cluster_lab='KM3'):
    cluster_scatter(data=data,x_lab='Recency',y_lab='MntTotal',cluster_lab=cluster_lab)
    box_plots(data,cluster_lab=cluster_lab,y="Recency")

def run_plots(data,cluster_lab='KM3'):

    print(f"--------------------Producing plots for {cluster_lab}--------------------")
    cluster_for_spent_and_len(data,cluster_lab='KM3')


    print("CLUSTERINS OVER DEMOGRAPHIC VARIABLES")
    cluster_for_demographic(data,cluster_lab=cluster_lab)

    print("EXAMINE DEALS AND CAMPAIGNS")
    box_plots(data,cluster_lab=cluster_lab,y="NumDealsPurchases")
    box_plots(data,cluster_lab=cluster_lab,y="NumDealsPurchasesNorm")

    print("CLUSTERS OVER CAMPAIGNS")
    campaign_bar_all(data, cluster_lab=cluster_lab)


    print("CLUSTERS OVER PURCHASE MEDIUM")
    cluster_for_medium(data,cluster_lab=cluster_lab)

    print("CLUSTERS OVER PRODUCT TYPE")
    cluster_for_type(data,cluster_lab=cluster_lab)

    print("CLUSTERS OVER RECENCY")
    cluster_scatter(data=data,x_lab='Recency',y_lab='MntTotal',cluster_lab='KM3')






