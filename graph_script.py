
from tkinter import Y
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#sns.set(font_scale=1.4)
#sns.set_context("poster", rc={"font.size":1.4,"axes.titlesize":8,"axes.labelsize":5})   
sns.set_context("paper", rc={"font.size":1.4,'axes.titlesize':18,
    'axes.labelsize':16,},)   
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.5)
sns.set(font_scale = 1.3)



data = pd.read_csv('./data/final_clustered.csv',index_col=0)


def box_plots(data,cluster_lab1="KM3",cluster_lab2="GM3",y="NumWebPurchases",y_lab=None):
    fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(14,10))

    sns.boxplot(data=data,ax=ax0,x=cluster_lab1, y=y,showfliers=False, showmeans=True,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}) #showfliers=False removes outliers in plot
    sns.boxplot(data=data,ax=ax1,x=cluster_lab2, y=y,showfliers=False, showmeans=True,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"})#,order=[2,1,0])

    sns.despine(offset=10, trim=True)
    ax0.set(xlabel=f'Cluster labels for {cluster_lab1}')
    ax1.set(xlabel=f'Cluster labels for {cluster_lab2}')

    #plt.title(f'Cluster labels for {y}')
    plt.suptitle(f'Cluster Labels Over {y_lab}',fontsize=18)

    plt.show()
    plt.clf() #clear figure


def bar_charts(data,cluster_lab1="KM3",cluster_lab2="GM3",y="Education",y_lab=None, reorder=False,catnames=['No','Yes']):
    fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(10,6))

    plt.figure(figsize=(9, 7))
    palette=sns.color_palette("Set2")
    if reorder:
        data[y] = data[y].astype('category')
        print(data[y].value_counts())
        y = data[y].cat.rename_categories(catnames)
    sns.countplot(data=data,ax=ax0,x=cluster_lab1,hue=y, palette = palette)
    sns.countplot(data=data,ax=ax1,x=cluster_lab2,hue=y, palette = palette)

    fig.suptitle(f"Cluster Profiles Over {y_lab}",fontsize=18)
    #plt.legend()
    plt.show()
    plt.clf()


# def cluster_scatter(data,x_lab,y_lab,cluster_lab):
#     #print(f'Scatter plots for {y_lab} vs {x_lab}')
#     plt.figure(figsize=(9, 7))
#     #palette=sns.color_palette("rocket", as_cmap=True)
#     #palette = sns.color_palette("cmo.balance", n_colors=64, desat=0.2)
#     plot = sns.scatterplot(data = data,x=data[x_lab], y=data[y_lab],hue=data[cluster_lab],alpha=.5)#,palette=palette)

#     plot.set_title(f"Cluster Profiles Over {x_lab} and {y_lab}")
#     #plt.legend()
#     plt.show()

def cluster_scatter(data,cluster_lab1="KM3",cluster_lab2="GM3",x='Income',y="MntTotal",x_lab=None, y_lab=None):
    fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(14,10))
    plt.figure(figsize=(9, 7))
 
    sns.scatterplot(data = data,ax=ax0,x=x, y=y,hue=cluster_lab1,alpha=.5)#,palette=palette)
    sns.scatterplot(data = data,ax=ax1,x=x, y=y,hue=cluster_lab2,alpha=.5,hue_order=['2','1','0'])#,palette=palette)

    fig.suptitle(f"Cluster Profiles Over {x_lab} and {y_lab}",fontsize=18)
    plt.show()
    plt.clf() #clear figure



def bar_charts_large(data,cluster_lab="KM3",y="Education",y_lab=None,y_lim=60):

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
    plt.title(f"Cluster Profiles Over {y_lab}")



# def campaign_bar(data,cluster_lab1="KM3",cluster_lab2="GM3",y="NumWebPurchases",campaign=None,y_lab=None):
#     fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(14,12))

#     # sns.countplot(data=data,ax=ax0,x=cluster_lab1,hue=y)#, palette = palette)
#     # sns.countplot(data=data,ax=ax1,x=cluster_lab2,hue=y)#, palette = palette)

#     sns.boxplot(data=data,ax=ax0,x=cluster_lab1, y=y)
#     sns.boxplot(data=data,ax=ax1,x=cluster_lab2, y=y)#,order=[2,1,0])

#     #plt.title(f"Cluster Profiles Based On Campaign {campaign[-1]} Acceptance")
#     fig.suptitle(f"Cluster Profiles Based On Campaign {campaign[-1]} Acceptance")
#     plt.show()
#     plt.clf()




def campaign_bar_all(data, cluster_lab1='KM3',cluster_lab2='GM3'):
    for campaign in [('AcceptedCmp1','Campaign 1 Acceptance'),('AcceptedCmp2','Campaign 2 Acceptance'),('AcceptedCmp3','Campaign 3 Acceptance'),
    ('AcceptedCmp4','Campaign 4 Acceptance'),('AcceptedCmp5','Campaign 5 Acceptance')]:
        bar_charts(data,cluster_lab1="KM3",cluster_lab2="GM3",y=campaign[0],y_lab=campaign[1], reorder=True)
        #campaign_bar(data,cluster_lab1,cluster_lab2,y="NumWebPurchases",campaign=campaign,y_lab=None)

    bar_charts(data,cluster_lab1="KM3",cluster_lab2="GM3",y='AcceptedCmpTot',y_lab='Total Campaign Acceptance')

def cluster_for_medium(data,cluster_lab1="KM3",cluster_lab2="GM3"):
    mediums = [('NumWebPurchases','Number of Web Purchases'),('NumWebPurchasesNorm','Number of Web Purchases (Normalized)'),
    ('NumCatalogPurchases',"Number of Catalog Purchases"),('NumCatalogPurchasesNorm','Number of Catalog Purchases (Normalized)'),
    ('NumStorePurchases','Number of Store Purchases'),('NumStorePurchasesNorm','Number of Store Purchases (Normalized)'),
    ('NumWebVisitsMonth','Number of Web Visits'),('NumWebVisitsMonthNorm','Number of Web Visits (Normalized)'),
    ('NumTotalPurchases','Number of Total Purchases'),('NumTotalPurchasesNorm','Number of Total Purchases (Normalized)')]
    for medium in mediums:
        box_plots(data,cluster_lab1,cluster_lab2,y=medium[0],y_lab=medium[1])


def cluster_for_type(data,cluster_lab1="KM3",cluster_lab2="GM3"):
    prod_types = [("MntWines",'Amount Spent on Wine'),("MntWinesNorm",'Amount Spent on Wine (Normalized)'),
    ("MntMeatProducts",'Amount Spent on Meat Products'),("MntMeatProductsNorm",'Amount Spent on Meat Products (Normalized)'),
    ("MntFishProducts",'Amount Spent on Fish Products'),("MntFishProductsNorm",'Amount Spent on Fish Products (Normalized)'),
    ("MntSweetProducts",'Amount Spent on Sweets'),("MntSweetProductsNorm",'Amount Spent on Sweets (Normalized)'),
                ("MntGoldProds",'Amount Spent on Gold Products'),("MntGoldProdsNorm",'Amount Spent on Gold Products (Normalized)')]
    for prod_type in prod_types:
        box_plots(data,cluster_lab1,cluster_lab2,y=prod_type[0],y_lab=prod_type[1])



def cluster_for_spent_and_len(data,cluster_lab1="KM3",cluster_lab2="GM3"):
    print("CLUSTERING RESULTS OVER AMOUNT SPENT AND LENGTH AS CUSTOMER")
    print("Income vs Total Amount Plot")
    cluster_scatter(data,cluster_lab1,cluster_lab2,x='Income',y="MntTotal",x_lab='Total Amount Spent', y_lab='Income')
    print("Income vs Total Normalized Amount Plot")
    cluster_scatter(data,cluster_lab1,cluster_lab2,x='Income',y="MntSpentNorm",x_lab='Total Amount Spent (Normalized)', y_lab='Income')
    #cluster_scatter(data,cluster_lab1,cluster_lab2,x='Income',y="MntSpentNorm",x_lab=None, y_lab='Normalized Total Amount Spent Vs. Income')


    box_plots(data,cluster_lab1,cluster_lab2,y='MntTotal', y_lab='Total Amount Spent')
    box_plots(data,cluster_lab1,cluster_lab2,y='MntSpentNorm',y_lab='Total Amount Spent (Normalized)')
    box_plots(data,cluster_lab1,cluster_lab2,y='Len_Customer',y_lab='Length as Customer')
    box_plots(data,cluster_lab1,cluster_lab2,y='SpendPropOfTotal',y_lab='Amount Spent as a Proportion of Total')
    box_plots(data,cluster_lab1,cluster_lab2,y='AvgPerPurchase', y_lab='Average Spent per Purchase')
    box_plots(data,cluster_lab1,cluster_lab2,y='NumTotalPurchasesNorm',y_lab='Total Number of Purchases (Normalized)')



    



def cluster_for_demographic(data,cluster_lab1="KM3",cluster_lab2="GM3"):
    # box_plots(data,cluster_lab=cluster_lab,y="Income")
    # box_plots(data,cluster_lab=cluster_lab,y="age")

    box_plots(data,cluster_lab1,cluster_lab2,y='Income', y_lab='Income')
    box_plots(data,cluster_lab1,cluster_lab2,y='age',y_lab='Age')

    for y in [('Education','Education'),('HasPartner','Having a Partner'),('NumChildren','Number of Children')]:
        for cluster in ['KM3','GM3']:
            bar_charts_large(data,cluster_lab=cluster,y=y[0],y_lab=y[1],y_lim=100)
            

def cluster_for_recency(data,cluster_lab1='KM3',cluster_lab2='GM3'):
    #cluster_scatter(data=data,x_lab='Recency',y_lab='MntTotal',cluster_lab=cluster_lab)
    cluster_scatter(data,cluster_lab1,cluster_lab2,x='Recency',y="MntTotal",x_lab='Recency', y_lab='Total Amount Spent')

    #box_plots(data,cluster_lab=cluster_lab,y="Recency")
    box_plots(data,cluster_lab1,cluster_lab2,y='Recency',y_lab='Recency')


def run_plots(data,cluster_lab1='KM3',cluster_lab2='GM3'):

    print(f"--------------------Producing plots for {cluster_lab1}--------------------")
    cluster_for_spent_and_len(data,cluster_lab1,cluster_lab2)


    print("CLUSTERINS OVER DEMOGRAPHIC VARIABLES")
    cluster_for_demographic(data,cluster_lab1,cluster_lab2)

    print("EXAMINE DEALS AND CAMPAIGNS")
    # box_plots(data,cluster_lab=cluster_lab,y="NumDealsPurchases")
    # box_plots(data,cluster_lab=cluster_lab,y="NumDealsPurchasesNorm")

    box_plots(data,cluster_lab1,cluster_lab2,y='NumDealsPurchases',y_lab='Number of Deal Purchases')
    box_plots(data,cluster_lab1,cluster_lab2,y='NumDealsPurchasesNorm',y_lab='Number of Deal Purchases (Normalized)')


    print("CLUSTERS OVER CAMPAIGNS")
    campaign_bar_all(data, cluster_lab1=cluster_lab1, cluster_lab2=cluster_lab2)


    print("CLUSTERS OVER PURCHASE MEDIUM")
    cluster_for_medium(data,cluster_lab1, cluster_lab2)

    print("CLUSTERS OVER PRODUCT TYPE")
    cluster_for_type(data,cluster_lab1,cluster_lab2)

    print("CLUSTERS OVER RECENCY")
    cluster_for_recency(data)

