# Customer-Segmentation
Please see the PowerPoint slides for main takeaways

This repository contains the K-means clustering analysis of the Kaggle dataset found [here](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) or [here](https://www.kaggle.com/datasets/jackdaoud/marketing-data?select=ifood_df.csv)




## Executive summary
- The goal of this analysis is to segment the store's customers in order to help the store maximize revenue and improve future campaign effectiveness.
- Cluster 5 are the recent high-income customers who value the in-store experience and a polished catalog.  They are not interested in deals, but will respond to a well-executed campaign.  When normalized by length spent as customers, this cluster brings the most revenue to the company.
- Cluster 1 are the loyalty high-income customers who are responsible for most historic revenue brought to the company.  Like Cluster 5, they value  in-store experience, a polished catalog, are not responsive to deals, but will respond to a well-structured campaign. 
- The store should concentrate on running well-structured campaigns to even better engage Cluster 1 and monitor the engagement of Cluster 5. A pleasant in-store shopping experience and a polished catalog are critical, but the store should study this demographic in detail to earn even more of their business.
- Cluster 0 are have higher than average income but more children (hence probably less dispensable income), are responsive to deals and an occasional campaign. It's best to target deals at these customers.
- Clusters 2-4 have lower than average income, accept deals at a higher rate, and visit the company's website.  The store can target deals and website promotions at this demographic.


## Detailed summary
### Cluster characteristics summary
- Clusters 1 and 5 are the high spenders with high income. Cluster 1 consists of loyalty customers but Cluster 5 customers spend the most per day. They are similar in most key dimensions, with Cluster 1 being slightly better educated, having more children, and having slightly higher likelihood of having a partner.
- Clusters 0-2 are the loyalty customers and Clusters 3-5 are the newer customers.
- Cluster 1 customers also have higher than average income, but probably have less disposable income due to having children
- Clusters 2-4 have lower income, are more likely to have children, and spend less

### Takeaways summary
- If run deals, target these at Clusters 2-4
- Concentrate on making/keeping the web page appealing for Clusters 0, 1, and 5
- Make sure to have a polished catalog, especially for Cluster 5 customers, who could bring in the most revenue in the future
- Make customers’ in-store experience customers pleasant as high-spending clusters seem to value it
- Learn from Campaign 2 in order to not repeat it
- Concentrate on running and improving campaigns like Campaign1 and 5, as these attract the high-spending customers
- Concentrate on identifying and attracting customers like Cluster 5 for highest future expected revenue

### Future work
- It would be helpful to have more information about the data set and the store in order to answer the following questions:
 - What do we know about the rationale behind each campaign? What distinguishes the campaigns?
 - What more can we learn about our customers? Specifically, are there factors that differentiate Cluster 5 that are not in the data?
 - What else can be learned about the way the store is making customers in-store and online shopping experience pleasant? Are there way to improve?
 - Can the store learn to do profitable business with Cluster 2-4?
 - Do A/B testing to judge the effectiveness of recommendations
 - Get more granular data on store purchases
 - Get more data on profitability rather than just revenue, as profitability is the key objective

