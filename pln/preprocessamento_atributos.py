import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Set
import gc
import numpy as np

class BagOfItems:
    def __init__(self, min_occur:int=0):

        #minimo de ocorrencias para um item ser considerado
        self.min_occur = min_occur

        """
        para cada item, haverá sua coluna correspondente
        dic_items mapeia o item para sua coluna
        """
        self.dic_items = {}
        self.set_cols_to_delete = set()

    def preprocess_text(self, df_data:pd.DataFrame, coluna:str):
        #Trata valores vazios (que, ao serem lidos, ficam como "NaN")
        for i,valor in enumerate(df_data[coluna]):
            if type(valor) != str:
                idx_item = df_data[coluna].index[i]
                df_data.at[idx_item, coluna] = ""

    def cria_bag_of_items(self, df:pd.DataFrame, arr_column_names:List[str],
                          remove_min_occur:bool =True) -> pd.DataFrame:

        if len(self.dic_items)==0:
            self.dic_items = self._cria_mapeamento_item_coluna(df , arr_column_names)
        #preprocessa os itens
        for coluna in arr_column_names:
            self.preprocess_text(df, coluna)
            
        #faz a matriz
        mat_bot,arr_ids = self._cria_matriz_bag_of_items(df, arr_column_names)

        #navega na coluna e identifica os itens que ocorreram menos que self.min_occur vezes
        set_cols_idx_to_delete = set()
        if remove_min_occur:
            for j in range(mat_bot.shape[1]):
                num_occur = mat_bot[:,j].sum()
                if num_occur < self.min_occur:
                    set_cols_idx_to_delete.add(j)

        #cria dataframe por meio da matriz e dos ids
        return self._cria_dataframe(mat_bot, arr_ids, set_cols_idx_to_delete)

    #aplica bag of itens já criado em um dataframe
    def aplica_bag_of_items(self,df:pd.DataFrame, arr_column_names:List[str]) -> pd.DataFrame:
        if len(self.dic_items) == 0:
            print("Ainda não foi criado o mapeamento do bag of items!")
            return None

        return self.cria_bag_of_items(df, arr_column_names, remove_min_occur=False)

    def _cria_mapeamento_item_coluna(self, df:pd.DataFrame , arr_column_names:List[str]) -> Dict[int,str]:
        dic_items = {}
        idx_item = 0
        for column_name in arr_column_names:
            for item in df[column_name]:
                if item not in dic_items:
                    dic_items[item] = idx_item
                    idx_item += 1

        return dic_items
    def _cria_matriz_bag_of_items(self, df:pd.DataFrame, arr_column_names:List[str]) -> np.ndarray:

        #essa matriz cada linha é uma instancia e cada coluna corresponde a ocorrencia de um item (0 ou 1)
        #cria uma matriz len(df) x idx_item para armazenar o "Bag of items"
        mat_bot = np.zeros((len(df),len(self.dic_items)),dtype=np.byte)

        #preenche a matriz com a presença de cada item
        arr_ids = []
        for i in range(len(df)):
            #adiciona o id do filme
            arr_ids.append(df.iloc[i]["id"])

            #navega nas colunas
            for column_name in arr_column_names:
                #obtem o item dessa instancia
                item_in_instance = df.iloc[i][column_name]
                if item_in_instance in self.dic_items:
                    #obtem a coluna correspondente a esse item
                    idx_item = self.dic_items[item_in_instance]
                    #define como presente este item nesta instancia
                    mat_bot[i][idx_item] += 1
            #print(mat_bot)
            if i%1000 == 0:
                print(f"{i}/{len(df)}")

        return mat_bot, arr_ids
    def _cria_dataframe(self, mat_bot:np.ndarray, arr_ids:List[int],
                            set_cols_idx_to_delete:Set[int]) -> pd.DataFrame:
        set_cols_name_to_delete = set()

        #cria os nomes das colunas a partir dos itens
        arr_columns = ["" for i in range(len(self.dic_items.keys()))]
        for item_name, item_idx in self.dic_items.items():
            if item_idx in set_cols_idx_to_delete:
                set_cols_name_to_delete.add(item_name)

            arr_columns[item_idx] = item_name


        #armazena e retorna o dataframe correspondente
        df_data = pd.DataFrame(mat_bot,columns=arr_columns)
        df_data["id"] = arr_ids
        #elimina as colunas desejadas
        if len(set_cols_name_to_delete)>0:
            self.set_cols_to_delete = set_cols_name_to_delete

        if len(self.set_cols_to_delete)>0:
            df_data.drop(self.set_cols_to_delete, axis = 1, inplace = True)


        return df_data

class BagOfWords:
    def __init__(self):
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        #max_df: remove palavras que ocorrem em mais que 90% dos documentos
        self.vectorizer = TfidfVectorizer(norm="l2",max_df=0.9)

    def preprocess_text(self, df_data:pd.DataFrame, coluna:str):
        #Trata valores vazios (que, ao serem lidos, ficam como "NaN")
        for i,valor in enumerate(df_data[coluna]):
            if type(valor) != str:
                idx_item = df_data[coluna].index[i]
                df_data.at[idx_item, coluna] = ""

    def cria_bow(self, df_data:pd.DataFrame, coluna:str) -> pd.DataFrame:
        self.preprocess_text(df_data, coluna)
        mat_bow = self.vectorizer.fit_transform(df_data[coluna])
        return pd.DataFrame(mat_bow.toarray(),columns=self.vectorizer.get_feature_names())

    def aplica_bow(self, df_data:pd.DataFrame, coluna:str) -> pd.DataFrame:
        self.preprocess_text(df_data, coluna)
        mat_bow = self.vectorizer.transform(df_data[coluna])
        return pd.DataFrame(mat_bow.toarray(),columns=self.vectorizer.get_feature_names())
