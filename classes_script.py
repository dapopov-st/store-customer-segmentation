#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys

np.random.seed(42)

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',500)

import hdbscan
import folium
import re

import plotly.express as px

from copy import deepcopy

import pickle
from pathlib import Path
model_path = Path('./models')

def write_to_file(model,model_name='',path=model_path):
    pickle.dump(model, open(Path.joinpath(model_path, model_name), 'wb'))



##############################################################################################################
class PcaAnalyzer:
    """
    Analyze and graph the PCA components
    Inputs:
        data: (df) preprocessed DataFrame
        subset: (list) of features pre-PCA (best to pass in SUBSET at top of the file)
        log_cols: (list) columns to log (best to pass in LOG_COLS at top of the file or have already logged columns in SUBSET)
        Scaler: type of scaler to use, such as StandardScaler, QuantileScaler, or RobustScaler
        pca_pct: (float) percentage of variance that must be explained by the PCA components
        
    Output:
        None: Call pca_explainer, pca_grapher, and get_select_components_df methods as needed
    Example:
        pca = PcaAnalyzer(data=df,subset=SUBSET,log_cols=LOG_COLS,Scaler=StandardScaler,pca_pct=.8)
        pca.pca_explainer()
        pca.pca_grapher(pca_indices=[1,2,3])
        pca.get_select_components_df(pca_indices=[1,2])

    """
    
    def __init__(self,data,subset=None,log_cols=[],Scaler=StandardScaler,pca_pct=None): 
        self.data = deepcopy(data)#.drop_duplicates() #XXX removing drop_duplicates. Causes issues since I am bringing in features without identifiers
        if not subset: #XXX changed default
            self.subset = list(self.data.columns)
        else:
            self.subset = subset #if len(subset) > 0 else self.get_nums(subset)
        self.type_checker()
        self.log_cols = log_cols
        self.scaler = Scaler()
        self.pca_pct=pca_pct #XXX changed here too, so that not forced to project initially and matches PCA default
        self.is_scaled, self.is_imputed, self.is_logged = False, False, False
        
        self.pca = self.get_pca()

    def type_checker(self): # If have additional ones, consider moving these to a utils.py file
        from pandas.api.types import is_numeric_dtype
        non_nums=set()
        for col in set(self.data[self.subset].columns):
            if not is_numeric_dtype(self.data[col]):
                non_nums.add(col)
        if len(non_nums)>0:
            raise ValueError(f"{non_nums} is/are not numeric. Please pass in a DataFrame of numeric types")



    def log_transform(self):
        for col in self.log_cols:
            self.data[col] = np.log1p(self.data[col])


    def scaled(self,X):
        """Returns a scaled version of features"""
        X_scaled=self.scaler.fit_transform(X)
        self.is_scaled=True
        return X_scaled


    def imputed(self,X):
      
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        X_imp=imp.fit_transform(X)
        return X_imp


    def get_pca(self):
        if self.log_cols: 
            self.log_transform()
        self.X = self.data[self.subset] # will use self.X later, add it as an attribute

        X_scaled=self.scaled(self.X)
        X_imputed = self.imputed(X_scaled)

        pca = PCA(n_components=self.pca_pct)

    
        pca_comps = pca.fit(X_imputed)
        self.N_pca = pca.n_components_
        print(f"There are {self.N_pca} components numbered 1 through {self.N_pca}")
        
        columns = ['PC'+str(i) for i in range(1,self.N_pca+1)]
        self.pca_df= pd.DataFrame(pca.fit_transform(X_imputed),columns=columns)
        
        #XXX adding explained_variance_ratio
        self.expl_var_ratio = pca.explained_variance_ratio_


        return pca_comps

    
    def pca_explainer(self):
        # Special thanks to https://www.reneshbedre.com/blog/principal-component-analysis.html#pca-loadings-plots
        # for code suggestions
        
        print(f"Proportion of variance explained by the {self.N_pca} PCA components (largest to smallest)")
        expl_var_ratio = self.pca.explained_variance_ratio_
        print(expl_var_ratio)

        print(f"Cumulative variance explained by the {self.N_pca} PCA components (largest to smallest)")
        print(np.cumsum(expl_var_ratio))

        loadings = self.pca.components_

        pc_list = ["PC"+str(i) for i in list(range(1, self.N_pca+1))]
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
        loadings_df['variable'] = self.X.columns.values
        loadings_df = loadings_df.set_index('variable')

        print("PCA Loadings")
        print(loadings_df)
        
        ###XXX changed figsize
        fig, ax = plt.subplots(figsize = (self.N_pca + 2,.4 * len(set(self.subset))))
        ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
        plt.show()

        print("Scree plot")


        columns = ['PCA'+str(i+1) for i in range(self.N_pca)]

        df_cluster_mnt_totals = pd.DataFrame([expl_var_ratio], columns = columns)

        df_cluster_mnt_totals=df_cluster_mnt_totals.T.reset_index().rename({0:'Percent of Variance Explained'},axis=1)

        fig=df_cluster_mnt_totals.sort_values('Percent of Variance Explained', ascending=False).plot.bar(color='#4503fc')

        fig.set_xticklabels(columns)
        
    def scree_plot(self, component = None):
        # XXX Added separate scree plot
        PC_values = np.arange(self.N_pca) + 1
        plt.plot(PC_values, self.expl_var_ratio, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        if component:
            plt.axhline(y = self.expl_var_ratio[component-1], linestyle = '--', color = 'red')
        plt.show()
    
    #XXX returns total variance explained by first <n> components
    def total_var(self,n):
        return self.expl_var_ratio[:n].sum()
        
    

    def validate_indices(self, pca_indices):
        """Validate the set of passed indices"""
        if not pca_indices: 
            raise ValueError("Please pass at least one PCA component")
       

        try:
            pca_indices = sorted(set(pca_indices))
        except:
            raise ValueError("Did you pass a set of sortable indices?")
        else:
            for idx in pca_indices:
                if type(idx) is not int:
                    raise ValueError("Indices must be integers")

            max_idx, min_idx=max(pca_indices), min(pca_indices)

            if max_idx > self.N_pca or min_idx < 1:
                raise ValueError(f"Did you pass a set of indices between 1 and {self.N_pca}?")
        


    def get_select_components(self,pca_indices):
        
        """
        Returns the PCA components 
        """
        self.validate_indices(pca_indices)
        return tuple([self.pca_df['PC'+str(x)] for x in pca_indices])

    def get_select_components_df(self,pca_indices):
        """
        Returns select components as a DataFrame
        """
        pca_comps=self.get_select_components(pca_indices)
        return pd.DataFrame(pca_comps).transpose()
      
    def pca_grapher(self,pca_indices=[]):
        """
        Ex: my_pca.pca_grapher(pca_indices=[1,2,4])
        """

        if len(pca_indices) != 3:
            raise ValueError("Please pass exactly 3 indices for PCA dimensions")

        self.x, self.y, self.z=self.get_select_components(pca_indices)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.x,self.y,self.z, c="maroon", marker="o",alpha=0.2 )
        ax.set_title(f"A 3D Projection Of Data In The Reduced Dimension for indices {pca_indices}")
        #XXX added axis labels
        ax.set_xlabel(f'Principal Component:{pca_indices[0]}')
        ax.set_ylabel(f'Principal Component: {pca_indices[1]}')
        ax.set_zlabel(f'Principal Component: {pca_indices[2]}')
        plt.show()

 ##############################################################################################################
class ClustererPcad:
    """
    Cluster the PCA component data using Algorithm and add labels to the data frame
    Inputs:
        data: (df) preprocessed DataFrame
        subset: (list) of features pre-PCA
        log_cols: (list) columns to log
        Scaler: type of scaler to use, such as StandardScaler, QuantileTransformer, or RobustScaler
        pca_pct: (float) percentage of variance that must be explained by the PCA components
        ClusterAlg: clustering algorithm, such as KMeans
        clstr_lbl: (str) label to be added to the data frame as Cluster+label with the cluster() method
        **cluster_kwargs: (dict) 
    Output:
        Clusterer() is callable and outputs a DataFrame with extra cluster label column  
    Examples:
    SUBSET = ['latitude', 'longitude', 'square_footage', 'lot_size', 'year_built',  'price']
    LOG_COLS=['square_footage', 'lot_size','price'] #Or pass in already logged columns in SUBSET
    PCA_PCT=.8

    #KMeans (default)
    from sklearn.cluster import KMeans
    clst_km = Clusterer(data=df,subset=SUBSET,log_cols=LOG_COLS,Scaler=StandardScaler,pca_pct=PCA_PCT)
    df_km=clst_km()

    #DBSCAN
    from sklearn.cluster import DBSCAN
    cluster_kwargs={'eps':0.30, 'min_samples':9}
    clst_db = Clusterer(data=df,ClusterAlg=DBSCAN(**cluster_kwargs),subset=SUBSET,log_cols=LOG_COLS,Scaler=RobustScaler,pca_pct=PCA_PCT,clstr_lbl='DB')
    df_db=clst_db()
  
    #GaussianMixture
    from sklearn.mixture import GaussianMixture
    cluster_kwargs={'n_components':3}
    clst_gm = ClustererPcad(pcas=pcas,ClusterAlg=GaussianMixture(**cluster_kwargs),clstr_lbl='GM')
    pcas_gm=clst_gm()

    #Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    cluster_kwargs={'n_clusters':5}
    ClusterAlg=AgglomerativeClustering(**cluster_kwargs)
    clst_ac = ClustererPcad(pcas=pcas,ClusterAlg=ClusterAlg,clstr_lbl='AC')
    pcas_ac=clst_ac()


    """
    
    def __init__(self,pcas,ClusterAlg=None,clstr_lbl=None,write=False, model_path='',model_name=''): 
        self.data = deepcopy(pcas)
        self.cluster_alg = ClusterAlg if ClusterAlg else KMeans(n_clusters=5)
        self.clstr_lbl = clstr_lbl if clstr_lbl else 'KM'

        #For saving model and writing it to file
        self.write=write
        self.model_path = model_path
        self.model_name = model_name
 


    def __call__(self):
        """
        Clusters the PCA components according to the specified algorithm and adds cluster labels to the original DataFrame
        """

        #Cluster the PCA components according to the clustering algorithm

        np.random.seed(42) #subtle issue: clustering algos don't have predict, 
                            #just fit and fit_predict, trying to ensure same seed gets used for both

        print('Fitting clusters...')
        self.cluster_alg.fit(self.data)
        if self.write:
            write_to_file(self.cluster_alg,model_name=self.model_name)
            

        yhat = self.cluster_alg.fit_predict(self.data)

        #Add cluster label to the original DataFrame and return the DataFrame with the added column
        print("Adding labels...")
        self.data["Clstrs"+self.clstr_lbl]= yhat


        print("Done")
        return self.data




##############################################################################################################
class GrapherPcad:
    """
    Plot the clusters after a clustering algorithm has been used to cluster the data
    Inputs:
        data: (df) DataFrame with an added column resulting from running Clusterer
        pca_pct: (float) percentage of variance that must be explained by the PCA components
        pca_indices: (list) indices for the select pca components, such as [1,2,3] or [1,3,4]. Must be of len. 3
        clstr_lbl: (str) label to be added to the data frame as Cluster+label with the cluster() method (eg, 'KM', don't include 'Clstrs')
        **cluster_kwargs: (dict) 
    Output:
        None, call plot_cluster and plot_interactive as needed
    Example:
        gr=Grapher(data=df_km,subset=SUBSET,pca_indices=[1,2,3],clstr_lbl='KM')
        gr.plot_clusters()
        gr.plot_interactive()

    TODO: Make Grapher work better with JUST the PCA components passed in

    """
    
    def __init__(self,pcas,clstr_lbl,pca_indices=[1,2,3]): 
        self.pcas = deepcopy(pcas)       
        self.pca_indices=pca_indices
        self.clstr_lbl = clstr_lbl
        self.check_label()
        self.x, self.y, self.z = None, None, None # add for lazy evaluation
        #self.pcayd = pcayd
    def check_label(self):
        if f'Clstrs{self.clstr_lbl}' not in set(self.pcas.columns):
            raise Exception((f"Clustering type with label {'Clstrs'+self.clstr_lbl} not added to the DataFrame."+ 
                "Please check your label name or make sure to call Clusterer class with the corresponding clustering algorithm"))


    def label_dimensions(self):
        self.pca_comps_df=pd.DataFrame(tuple([self.pcas['PC'+str(x)] for x in self.pca_indices])).transpose()
        self.pcas['x'], self.pcas['y'], self.pcas['z'] = self.pca_comps_df.iloc[:,0], self.pca_comps_df.iloc[:,1],self.pca_comps_df.iloc[:,2]

     
        self.pca_comps_df["Clstrs"+self.clstr_lbl] = self.pcas["Clstrs"+self.clstr_lbl]

        self.cluster_nums = sorted(self.pca_comps_df['Clstrs'+self.clstr_lbl].unique())
        print(f"{self.pca_comps_df['Clstrs'+self.clstr_lbl].nunique()} clusters with labels in {self.cluster_nums}")

    
    def plot_clusters(self, fltr_clstr_num=None):
        """
        clstr_lbl: cluster label obtained by running the Clusterer class with the corresponding clustering algorithm
        """
        self.label_dimensions()
       
        from matplotlib import colors
        from matplotlib.colors import ListedColormap
        cmap = colors.ListedColormap(['#9467bd',"#B9C0C9", "#D6B2B1", "#682F2F", "#9E726F", "#9F8A78", "#F3AB60",
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(111, projection='3d', label="Clstrs"+self.clstr_lbl)
        ax.set_xlabel(f'Principal Component:{self.pca_indices[0]}')
        ax.set_ylabel(f'Principal Component:{self.pca_indices[1]}')
        ax.set_zlabel(f'Principal Component:{self.pca_indices[2]}')
        ax.scatter(self.pcas['x'], self.pcas['y'], self.pcas['z'], s=40, c=self.pca_comps_df["Clstrs"+self.clstr_lbl], marker='o', cmap = cmap )
        ax.set_title("The Plot Of The Clusters")
        plt.show()

    def plot_interactive(self):
        labels = {'x':f'PC{self.pca_indices[0]}','y':f'PC{self.pca_indices[1]}','z':f'PC{self.pca_indices[2]}'}
        fig = px.scatter_3d(self.pcas, x='x', y='y', z='z',
                    color="Clstrs"+self.clstr_lbl,opacity=0.2,labels=labels)
        fig.show()



##############################################################################################################

# class Clusterer:
#     """
#     Cluster the PCA component data using Algorithm and add labels to the data frame
#     Inputs:
#         data: (df) preprocessed DataFrame
#         subset: (list) of features pre-PCA
#         log_cols: (list) columns to log
#         Scaler: type of scaler to use, such as StandardScaler, QuantileTransformer, or RobustScaler
#         pca_pct: (float) percentage of variance that must be explained by the PCA components
#         ClusterAlg: clustering algorithm, such as KMeans
#         clstr_lbl: (str) label to be added to the data frame as Cluster+label with the cluster() method
#         **cluster_kwargs: (dict) 
#     Output:
#         Clusterer() is callable and outputs a DataFrame with extra cluster label column  
#     Examples:
#     SUBSET = ['latitude', 'longitude', 'square_footage', 'lot_size', 'year_built',  'price']
#     LOG_COLS=['square_footage', 'lot_size','price'] #Or pass in already logged columns in SUBSET
#     PCA_PCT=.8

#     #KMeans (default)
#     from sklearn.cluster import KMeans
#     clst_km = Clusterer(data=df,subset=SUBSET,log_cols=LOG_COLS,Scaler=StandardScaler,pca_pct=PCA_PCT)
#     df_km=clst_km()

#     #DBSCAN
#     from sklearn.cluster import DBSCAN
#     cluster_kwargs={'eps':0.30, 'min_samples':9}
#     clst_db = Clusterer(data=df,ClusterAlg=DBSCAN(**cluster_kwargs),subset=SUBSET,log_cols=LOG_COLS,Scaler=RobustScaler,pca_pct=PCA_PCT,clstr_lbl='DB')
#     df_db=clst_db()
  
#     #GaussianMixture
#     from sklearn.mixture import GaussianMixture
#     cluster_kwargs={'n_components':5}
#     clst_gm = Clusterer(data=df,ClusterAlg=GaussianMixture(**cluster_kwargs),subset=SUBSET,log_cols=LOG_COLS,Scaler=RobustScaler,pca_pct=PCA_PCT,clstr_lbl='GM')
#     df_gm=clst_gm()

#     #Agglomerative Clustering
#     from sklearn.cluster import AgglomerativeClustering
#     clst_ac = Clusterer(data=df,ClusterAlg=AgglomerativeClustering(**cluster_kwargs),subset=SUBSET,log_cols=LOG_COLS,pca_pct=PCA_PCT,clstr_lbl='AC')
#     df_ac=clst_ac()


#     """
    
#     def __init__(self,data,ClusterAlg=None,subset=[],log_cols=[],Scaler=None,pca_pct=None,clstr_lbl=None,drop_dupl=False,impute=True,write=False,
#     model_path='',model_name='',df_name=''): 
#         self.data = deepcopy(data).drop_duplicates() if drop_dupl else deepcopy(data)
#         self.subset = subset if len(subset)>0 else self.data.columns
#         self.type_checker()
#         self.log_cols = log_cols
#         self.scaler = Scaler() if Scaler else StandardScaler()
#         self.pca_pct=pca_pct if pca_pct else 0.8
#         self.cluster_alg = ClusterAlg if ClusterAlg else KMeans(n_clusters=5)
#         self.clstr_lbl = clstr_lbl if clstr_lbl else 'KM'
#         self.impute = impute
#         self.is_scaled, self.is_imputed, self.is_logged = False, False, False
#         #For saving model and writing it to file
#         self.write=write
#         self.model_path = model_path
#         self.model_name = model_name
 
#     def type_checker(self): # If have additional ones, consider moving these to a utils.py file
#         from pandas.api.types import is_numeric_dtype
#         non_nums=set()
#         for col in set(self.data[self.subset].columns):
#             if not is_numeric_dtype(self.data[col]):
#                 non_nums.add(col)
#         if len(non_nums)>0:
#             raise ValueError(f"{non_nums} is/are not numeric. Please pass in a DataFrame of numeric types")

#     def log_transform(self):
#         for col in self.log_cols:
#             self.data[col] = np.log1p(self.data[col])
#         self.is_logged = True


#     def scaled(self,X):
#         """Returns a scaled version of features, where the type of scaler must be passed to the class constructor"""
#         X_scaled=self.scaler.fit_transform(X)
#         self.is_scaled=True
#         return X_scaled


#     def imputed(self,X):
#         """
#         The simplest imputation strategy was used to keep the code general.If a more sophisticated imputation strategy is desired, 
#         impute the missing values before passing in the data and this method will be a no-op.  In addition, since clustering is 
#         usually performed on numeric/ordinal values, this assumes that missing values are np.nan's
#         """
#         if not self.is_scaled: #soft warnings in case values have been scaled prior to passing in data
#             print("Warning: Your data has not been scaled.  Make sure you've scaled it before passing it in.")
#         #Choose to import this here rather than at the top of the file, as conventional, to avoid passing extra
#         # instructions to the class user.
#         from sklearn.impute import SimpleImputer 
#         if self.impute:
#             imp = SimpleImputer(missing_values=np.nan, strategy='median')
#             X_imp=imp.fit_transform(X)
#             self.is_imputed=True
#             return X_imp
#         else:
#             return X


#     def get_pca(self,X):
#         if not self.is_imputed: #soft warnings in case values have been imputed prior to passing in data
#             print("Warning: Missing values have not been imputed.  Make sure you've imputed them before passing in the data.")
#         if self.pca_pct == 1:
#             pca = X
#             return X
#         else:
#             pca = PCA(n_components=self.pca_pct)
        
#             pca_comps = pca.fit_transform(X)

#             N_pca = pca.n_components_
#             print(f"{N_pca} components chosen")
#             columns = ['col'+str(i) for i in range(1,N_pca+1)]

#             #Make a DataFrame of PCA components
#             pca_comps= pd.DataFrame(pca_comps,columns=columns)
#             return pca_comps

    


#     def processed_pcas(self):
#         #Run this line here rather than in __init__ to get log-transformed features
#         X = self.data[self.subset] 

#         # Scale X and impute missing values
#         X_scaled=self.scaled(X)
#         X_imputed = self.imputed(X_scaled)

#         # Get PCA components
#         pca_comps= self.get_pca(X_imputed) 
#         return pca_comps


#     def predict(self,X):
#         return self.labels_.astype(int)
#     def __call__(self):
#         """
#         Clusters the PCA components according to the specified algorithm and adds cluster labels to the original DataFrame
#         """
#         #Log-transform skewed features
#         #If chose some columns to which to apply log transforms, ie, log_cols is nonempty
#         if self.log_cols: 
#             self.log_transform()


#         print("Getting PCAs...")
#         #Get PCA components from features
#         pca_comps = self.processed_pcas()

#         print(f"Logged: {self.is_logged}. Scaled: {self.is_scaled}. Imputed: {self.is_imputed}.")

#         #Cluster the PCA components according to the clustering algorithm

#         np.random.seed(42) #subtle issue: clustering algos don't have predict, 
#                             #just fit and fit_predict, trying to ensure same seed gets used for both

#         print('Fitting clusters...')
#         self.cluster_alg.fit(pca_comps)
#         if self.write:
#             write_to_file(self.cluster_alg,model_name=self.model_name)
            

#         yhat = self.cluster_alg.fit_predict(pca_comps)

#         #Add cluster label to the original DataFrame and return the DataFrame with the added column
#         print("Adding labels...")
#         self.data["Clstrs"+self.clstr_lbl]= yhat


#         print("Done")
#         return self.data


        
# # ##############################################################################################################

# # class Grapher:
# #     """
# #     Plot the clusters after a clustering algorithm has been used to cluster the data
# #     Inputs:
# #         data: (df) DataFrame with an added column resulting from running Clusterer
# #         pca_pct: (float) percentage of variance that must be explained by the PCA components
# #         pca_indices: (list) indices for the select pca components, such as [1,2,3] or [1,3,4]. Must be of len. 3
# #         clstr_lbl: (str) label to be added to the data frame as Cluster+label with the cluster() method (eg, 'KM', don't include 'Clstrs')
# #         **cluster_kwargs: (dict) 
# #     Output:
# #         None, call plot_cluster and plot_interactive as needed
# #     Example:
# #         gr=Grapher(data=df_km,subset=SUBSET,pca_indices=[1,2,3],clstr_lbl='KM')
# #         gr.plot_clusters()
# #         gr.plot_interactive()

# #     TODO: Make Grapher work better with JUST the PCA components passed in

# #     """
    
# #     def __init__(self,data,clstr_lbl,subset=[],log_cols=[],Scaler=None,pca_pct=0.8,pca_indices=[1,2,3]): 
# #         self.data = deepcopy(data).drop_duplicates()
# #         #if subset:
# #         self.subset = subset if len(subset)>0 else self.data.columns
       

# #         self.log_cols=log_cols
# #         self.scaler = Scaler() if Scaler else StandardScaler()
# #         self.pca_pct=pca_pct
# #         self.pca_indices=pca_indices
# #         self.clstr_lbl = clstr_lbl
# #         self.x, self.y, self.z = None, None, None # add for lazy evaluation
# #         #self.pcayd = pcayd

# #     ((f"Clustering type with label {'Clstrs'+self.clstr_lbl} not added to the DataFrame."+ 
# #             "Please check your label name or make sure to call Clusterer class with the corresponding clustering algorithm"))
# #     def plot_clusters(self, fltr_clstr_num=None):
# #         """
# #         clstr_lbl: cluster label obtained by running the Clusterer class with the corresponding clustering algorithm
# #         """
# #         #if not self.pcayd:
# #         pca = PcaAnalyzer(data=self.data,subset=self.subset,log_cols=self.log_cols,Scaler=StandardScaler,pca_pct=self.pca_pct)
# #         # else:
# #         #     pca = PcaAnalyzer(data=self.data,subset=self.subset,log_cols=self.log_cols,Scaler=StandardScaler,pca_pct=1)
# #         #Add pca component labels to the original DataFrame
# #         self.data['x'], self.data['y'], self.data['z'] = pca.get_select_components(self.pca_indices)


# #         #Add cluster labels to PCA components

# #         if ('Clstrs'+self.clstr_lbl) not in self.data.columns:
# #             raise NameError
# #         self.pca_comps_df= pd.DataFrame()
# #         #Add cluster label to pca components df
# #         self.pca_comps_df["Clstrs"+self.clstr_lbl] = self.data["Clstrs"+self.clstr_lbl]

# #         self.cluster_nums = sorted(self.pca_comps_df['Clstrs'+self.clstr_lbl].unique())
# #         print(f"{self.pca_comps_df['Clstrs'+self.clstr_lbl].nunique()} clusters with labels in {self.cluster_nums}")

    

# #         #Set matplotlib params, possibly rewrite this if need more colors
# #         from matplotlib.colors import ListedColormap
# #         cmap = colors.ListedColormap(['#9467bd',"#B9C0C9", "#D6B2B1", "#682F2F", "#9E726F", "#9F8A78", "#F3AB60",
# #         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# #         fig = plt.figure(figsize=(10,8))
# #         ax = plt.subplot(111, projection='3d', label="Clstrs"+self.clstr_lbl)



# #         if fltr_clstr_num in self.cluster_nums:
# #             self.specific_cluster_subroutine(fltr_clstr_num)
   

# #         if fltr_clstr_num not in self.cluster_nums: #None would not be in the set
# #             ax.scatter(self.data['x'], self.data['y'], self.data['z'], s=40, c=self.pca_comps_df["Clstrs"+self.clstr_lbl], marker='o', cmap = cmap )
# #         else:
# #             ax.scatter(self.masked_data['x'], self.masked_data['y'], self.masked_data['z'], s=40, c=self.masked_pca_comps_df["Clstrs"+self.clstr_lbl], marker='o', cmap = cmap )

# #         ax.set_title("The Plot Of The Clusters")
# #         plt.show()

# # def specific_cluster_subroutine(self, cluster_num):

# #     #if cluster_num in self.cluster_nums: # a bit inefficient to filter cluster after assigning all clusters, but runs fast and requires fewest code modifications
# #     print(f"Plotting for the chosen cluster {cluster_num}")
# #     mask_pca = (self.pca_comps_df['Clstrs'+self.clstr_lbl] == cluster_num)
# #     self.masked_pca_comps_df = self.pca_comps_df[mask_pca]
# #     mask_data = (self.data['Clstrs'+self.clstr_lbl] == cluster_num)
# #     self.masked_data = self.data[mask_pca]



# # def plot_interactive(self):
# #     fig = px.scatter_3d(self.data, x='x', y='y', z='z',
# #                 color="Clstrs"+self.clstr_lbl,opacity=0.2)
# #     fig.show()

# # def plot_specific_cluster(self,cluster_num):
# #     if cluster_num not in set(self.cluster_nums):
# #         raise ValueError("Cluster number chosen is not one of the cluster numbers assigned")
# #     self.plot_clusters(fltr_clstr_num=cluster_num)