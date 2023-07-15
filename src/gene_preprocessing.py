'''
Utilidades para selección de características, genes, etc
'''
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, TransformerMixin


class SelectByPValue(BaseEstimator, TransformerMixin):
    def __init__(self, thresh = 0.05, scaled = True):
        self.thresh = thresh
        self.scaled = scaled

    def fit(self, X, y=None):
        if type(X)==pd.core.frame.DataFrame:
            self.columns_ = X.columns
        self.f_vals, self.p_vals = f_classif(X, y)
        return self

    def transform(self, X, y=None):
        if self.scaled:
            indexs = np.where(self.p_vals<self.thresh/X.shape[1])[0]
        else:
            indexs = np.where(self.p_vals<self.thresh)[0]
        
        if type(X)==pd.core.frame.DataFrame:
            self.columns_ = X.columns
            return X.iloc[:,indexs]

        else:
            return X[:, indexs]
    
    def get_feature_names_out(self):
        pass

class SelectByIterRF(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=20, thresh=0.95, pos_label='Solid Tissue Normal'):
        self.thresh = thresh
        self.max_iter = max_iter
        self.pos_label = pos_label
        self.selected_features = None

    def fit(self, X:pd.DataFrame, y=None):
       
        iter = 0
        last_f1 = 1

        obtained_features = []
        while (iter<self.max_iter) and (last_f1>self.thresh):

          x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state = 42)

          selector = SelectFromModel(RandomForestClassifier(random_state=42))
          selector.fit(x_train, y_train)
          clf = selector.estimator_

          last_f1 = f1_score(y_test, clf.predict(x_test), pos_label= self.pos_label)
          iter_features = list(selector.get_feature_names_out())
          obtained_features += iter_features
          iter+=1

          X = X.drop(columns = iter_features)

        self.selected_features = obtained_features
        return self

    def transform(self, X:pd.DataFrame, y=None):
        return X[self.selected_features]
    
    def get_feature_names_out(self):
        pass
    
def get_relevant_cpg(df, max_iter = 20, thresh = 0.95):
  '''
  Algoritmo de selección basado en paper de algoritmo BIGBIOCL
  '''

  x, y = df.drop(columns=['sample_type']), df['sample_type']


  iter = 0
  last_f1 = 1

  obtained_features = []
  while (iter<max_iter) and (last_f1>thresh):

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state = 42)

    selector = SelectFromModel(RandomForestClassifier(random_state=42))
    selector.fit(x_train, y_train)
    clf = selector.estimator_

    last_f1 = f1_score(y_test, clf.predict(x_test), pos_label= 'Solid Tissue Normal')
    iter_features = list(selector.get_feature_names_out())
    obtained_features += iter_features
    iter+=1

    x = x.drop(columns = iter_features)

  return obtained_features
    
def get_gene_df(df: pd.DataFrame, mapping):
  '''
  Retorna df con metilación promedio de genes.
    - df: dataframe con sitios CpG
    - mapping: dataframe con columnas UCSC_RefGene_Name (nombre del gen) y IlmnID (lista de sitios CpG)
  '''
  gene_df = pd.DataFrame()
  columns = []
  mean_cpgs = []
  for _, row in mapping.iterrows():
    columns.append(row['UCSC_RefGene_Name'])
    mean_cpgs.append(df.loc[:, df.columns.isin(row['IlmnID'])].mean(axis=1))
  gene_df = pd.concat(mean_cpgs, axis=1)
  gene_df.columns = columns
  return gene_df

