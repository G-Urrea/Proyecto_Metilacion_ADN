'''
Funciones asociadas a la consulta a API, descarga de datos y preprocesameinto básico.
'''

import gdown
import pandas as pd
import requests
import json
# Para evitar warnings
pd.options.mode.chained_assignment = None

def download_files_from_gd(files_dict):
  '''
  Función para descargar archivos desde google drive
  - files_dict: Diccionario en formato {filename_1: drive_id_1, ...., filename_n:drive_id_n}
  '''
  drive_prefix = "https://drive.google.com/uc?id="
  for file in files_dict:
    gdown.download(f"{drive_prefix}{files_dict[file]}", file)

def get_gdc_files_data(in_filters, additional_fields, query_size=2000):
  '''
  Obtención de archivos desde la API de la GDC.

  - in_filters: 
  '''
  files_endpt = "https://api.gdc.cancer.gov/files"
  filters_list = []
  for field in in_filters:
    query = {
        "op":"in",
        "content":{
            "field": field,
            "value": [in_filters[field]]
        }
    }
    filters_list.append(query)

  filters = {
      "op": "and",
      "content":filters_list
  }

  query_fields = list(in_filters.keys()) + additional_fields
  params = {
      "filters": json.dumps(filters),
      "fields": ",".join(query_fields),
      "format": "JSON",
      "size": str(query_size)
      }
  response = requests.get(files_endpt, params = params)
  return response

def download_gdc_data(filename, ids_list):
  data_endpt = "https://api.gdc.cancer.gov/data"

  params = {"ids": ids_list}
  response = requests.post(data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})

  with open(f'{filename}.tar.gz', "wb") as output_file:
      output_file.write(response.content)

def gdc_response_to_df(response):
  df_list = []
  # Obtener datos y guardarlos en lista de diccionarios
  for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
      file_id = file_entry["file_id"]
      file_size = int(file_entry['file_size'])
      sample_type = file_entry['cases'][0]['samples'][0]['sample_type']
      disease_type = file_entry['cases'][0]['disease_type']
      df_list.append({'file_id':file_id, 'size':file_size, 'disease':disease_type, 'sample_type':sample_type})

def concatenate_gdc_files(base_folder):
  list_df = []
  # Al descomprimir el archivo se generarán un montón de carpetas con archivos de texto
  # Este código lee los archivos, los convierte en dataframes adecuados y los almacena
  for dir in os.listdir(base_folder):
    id = dir
    dir_path = f"/{base_folder}/{dir}"
    if os.path.isdir(dir_path):
      for path in os.scandir(dir_path):
        if path.is_file():
          path_to = f"{dir_path}/{path.name}"
          df_temp = pd.read_csv(path_to, header=None , sep='\t')

          df_temp = df_temp.T
          new_header = df_temp.iloc[0] #grab the first row for the header
          df_temp = df_temp[1:] #take the data less the header row
          df_temp.columns = new_header #set the header row as the df header
          df_temp['file_id'] = id
          list_df.append(df_temp)
  # Concatenar dataframes
  df_samples = pd.concat(list_df, ignore_index=True)
  return df_samples
