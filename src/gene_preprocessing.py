import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


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