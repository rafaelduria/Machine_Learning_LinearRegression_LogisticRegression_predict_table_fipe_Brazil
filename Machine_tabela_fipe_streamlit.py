import warnings
import streamlit as st
import plotly.express as px
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

#Lendo Base de dados
@st.cache_data
def busca_cache(suppress_st_warning=True):
    base = pd.read_csv("https://raw.githubusercontent.com/rafaelduria/Machine_Learning_LinearRegression_LogisticRegression_predict_table_fipe_Brazil/main/tabela_fipe_historico_precos.csv", sep=',')
    return base
    
base = busca_cache()

base.drop(['Unnamed: 0'], axis=1, inplace=True)
base['marca'] = base['marca'].str.upper()
base['modelo'] = base['modelo'].str.upper()
base['Drop'] = base['anoModelo'] - base['anoReferencia']
base = base[(base.Drop <0)]
#excluindo coluna
base.drop(['Drop'], axis=1, inplace=True)
#convertendo para inteiro
base[['anoModelo','mesReferencia','anoReferencia']] = base[['anoModelo','mesReferencia','anoReferencia']].astype(int)
#ordenando colunas ano e mes
base = base.sort_values(by=['anoReferencia' ,'mesReferencia'], ignore_index = True, ascending = True)
#Concatenando colunas "/"
base['Data'] = base['mesReferencia'].map(str) + '/' + base  ['anoReferencia'].map(str)
#Convertendo para data
base['Data'] = pd.to_datetime(base['Data'], dayfirst=True)
#ordenando data
base['Data'] = base['Data'].dt.strftime('%m/%Y')
#Concatenando colunas "/"
base['anoModelo'] =  '01/01/' + base  ['anoModelo'].map(str)
base['anoModelo'] = pd.to_datetime(base['anoModelo'], dayfirst=True)
base['anoModelo'] = base['anoModelo'].dt.strftime('%Y')

#.\env\Scripts\activate.ps1
# streamlit run FIPE.py
#Base para Machine Learning


#deixar página tamanho grande
st.set_page_config(layout='wide')
#Titulo
st.header('Tabela Fipe')

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)


#Barra lateral esquerda
#titulo
st.sidebar.title('Filtro')
#lista marca carros tabela fipe
Lista_Fabricantes = list(base['marca'].unique())
#ordenando lista Lista_Fabricantes
Lista_Fabricantes = sorted(Lista_Fabricantes)
#botao radio, na lateral estquerda "sidebar"
Fabricante_Escolhido = st.sidebar.selectbox(label = 'Fabricantes', options = Lista_Fabricantes, index=49)
#filtrando base de base conforme filtro  Fabricante_Escolhido
base = base.loc[(base['marca'] == Fabricante_Escolhido)]


#lista modelos carros tabela fipe
Lista_Modelos = list(base['modelo'].unique())
#ordenando lista Lista_Modelos
Lista_Modelos = sorted(Lista_Modelos)
Modelo_Escolhido = st.sidebar.selectbox(label = 'Modelo', options = Lista_Modelos)
#filtrando base de base conforme filtro  Fabricante_Escolhido
base = base.loc[(base['modelo'] == Modelo_Escolhido)]


#lista modelos carros tabela fipe
Lista_anoModelo = list(base['anoModelo'].unique())
#ordenando lista Lista_Modelos
Lista_anoModelo = sorted(Lista_anoModelo)
anoModelo_Escolhido = st.sidebar.selectbox(label = 'Ano Modelo', options = Lista_anoModelo)
#filtrando base de base conforme filtro  Fabricante_Escolhido
base = base.loc[(base['anoModelo'] == anoModelo_Escolhido)]


st.subheader('Dados Histórico')
options = st.multiselect(label='', options = list(base.Data), default = list(base.Data)[-8:])
base = base[base['Data'].isin(options)]
grafico = px.line(base, x=base['Data'], y=base['valor'],text=base.valor)
grafico = grafico.update_traces(textposition="top center")
#respeitar tamanho
st.plotly_chart(grafico, use_container_width=True)

Dataset = base.loc[:,['mesReferencia','anoReferencia','valor']]

if len(Dataset)>2:
  X = Dataset[Dataset.columns[0:2]].values
  y = Dataset.loc[:,'valor'].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  modelo = RandomForestRegressor(max_depth=30, n_estimators=200)
  modelo = modelo.fit(X_train, y_train)



  calendario_lista_prever = [
              ['9', '2022'],
              ['10', '2022'],
              ['11', '2022'],
              ['12', '2022'],
              ['1', '2023'],
              ['2', '2023'],
              ['3', '2023'],
              ['4', '2023'],
              ['5', '2023'],
              ['6', '2023'],
              ['7', '2023'],
              ['8', '2023'],
              ['9', '2023'],
              ['10', '2023'],
              ['11', '2023'],
              ['12', '2023'],
              ['1', '2024']
              ]


  Valores_previstos_lista=[]
  for i in calendario_lista_prever:
      Mês_Referencia=i[0]
      Ano_Referencia=i[1]
      entrada=[[Mês_Referencia, Ano_Referencia]]
      temp = modelo.predict(entrada)[0]
      Valores_previstos_lista.append(temp)

  #convertendo listas para dataframe
  previsao_calendario = pd.DataFrame(calendario_lista_prever, columns=['mesReferencia', 'anoReferencia'])
  #convertendo listas para dataframe
  Valores_previstos_dataframe = pd.DataFrame(Valores_previstos_lista, columns=['Valores_Previstos'])
  #convertendo valores previsto para inteiros
  Valores_previstos_dataframe[['Valores_Previstos']] = Valores_previstos_dataframe[['Valores_Previstos']].astype(int)
  #jutando dataframes
  previsao_calendario = pd.merge(previsao_calendario, Valores_previstos_dataframe, left_index=True, right_index=True)


  previsao_calendario['Data'] = previsao_calendario['mesReferencia'].map(str) + '/' + previsao_calendario  ['anoReferencia'].map(str)
  #Convertendo para data
  previsao_calendario['Data'] = pd.to_datetime(previsao_calendario['Data'], dayfirst=True)
  #ordenando data
  previsao_calendario['Data'] = previsao_calendario['Data'].dt.strftime('%m/%Y')



  st.subheader('Previsão')
  y_previsto = modelo.predict(X_test)
  Quanto_linha = ('R² = {} Quanto a linha de regressao ajusta-se aos dados'.format(modelo.score(X_train, y_train).round(2)))
  ### Gerando previsões para os dados de TESTE (X_test) utilizando o método *predict()* do objeto "modelo"
  st.write(Quanto_linha)
  ### Obtendo o coeficiente de determinação (R²) para as previsões do nosso modelo
  try:
      R2_ = 'R² = %s Previsões do nosso modelo' % metrics.r2_score(y_test, y_previsto).round(2)
  except:
      R2_ = str((metrics.r2_score(y_test, y_previsto)))

  if R2_ == 'nan':
    st.write('Não foi possível fazer previsão poucos dados')
  else:
    st.write(R2_)

  grafico_previsao = px.line(previsao_calendario,x=previsao_calendario['Data'],y=previsao_calendario['Valores_Previstos'],text=previsao_calendario.Valores_Previstos)
  grafico_previsao = grafico_previsao.update_traces(textposition="top center")
  st.plotly_chart(grafico_previsao, use_container_width=True) #respeitar tamanho
else:
  st.header('Não foi possível fazer previsão poucos dados')
