import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from yellowbrick.regressor import PredictionError

st.session_state.selected_model = st.session_state.get('selected_model', 'Linear Regression')

# Carregando o Dataset
df = pd.read_csv('car_resale_prices.csv')

# Tratando o Dataset
df.drop(columns=['Unnamed: 0'], inplace=True)
df['registered_year'] = df['registered_year'].fillna(df['full_name'].apply(lambda x: x.split()[0]))
df.loc[df['insurance'] == 'Not Available', 'insurance'] = np.nan
df.loc[df['insurance'] == '1', 'insurance'] = np.nan
df.loc[df['insurance'] == '2', 'insurance'] = np.nan
df.loc[df['insurance'] == 'Third Party', 'insurance'] = 'Third Party insurance'
df.dropna(inplace=True)

# Conversão da coluna mileage
def converter_coluna_mileage(x):
    if 'kmpl' in x:
        return float(x.replace('kmpl', '').strip())
    elif 'km/kg' in x:
        km_kg = float(x.replace('km/kg', '').strip())
        return km_kg / 1.39 # pesquisei a conversão de gás natural para quilômetros por litro
    
df['mileage'] = df['mileage'].apply(converter_coluna_mileage)
df.rename(columns={'mileage': 'mileage_kmpl'}, inplace=True)

# Conversão da coluna kms_driven
def converter_coluna_kms_driven(x):
    return float(x.replace(',', '').replace('Kms', '').strip())
    
df['kms_driven'] = df['kms_driven'].apply(converter_coluna_kms_driven)

# Conversão da coluna engine_capacity
def converter_coluna_engine_capacity(x):
    if pd.isna(x):
        return x
    else:
        return int(x.replace('cc', '').strip())
    
df['engine_capacity'] = df['engine_capacity'].apply(converter_coluna_engine_capacity)
df.rename(columns={'engine_capacity': 'engine_capacity_cc'}, inplace=True)
df.loc[(df['fuel_type'] == 'Electric') & (df['engine_capacity_cc'].isin([0, 72])), 'engine_capacity_cc'] = np.nan
df.dropna(inplace=True)

# Conversão da coluna registered_year
def converter_coluna_registered_year(x):
    if 'Jan' in x:
        return int(x.replace('Jan', '').strip())
    elif 'Feb' in x:
        return int(x.replace('Feb', '').strip())
    elif 'Mar' in x:
        return int(x.replace('Mar', '').strip())
    elif 'Apr' in x:
        return int(x.replace('Apr', '').strip())
    elif 'May' in x: 
        return int(x.replace('May', '').strip())
    elif 'Jun' in x: 
        return int(x.replace('Jun', '').strip())
    elif 'Jul' in x:
        return int(x.replace('Jul', '').strip()) 
    elif 'Aug' in x: 
        return int(x.replace('Aug', '').strip())
    elif 'Sept' in x: 
        return int(x.replace('Sept', '').strip())
    elif 'Oct' in x: 
        return int(x.replace('Oct', '').strip())
    elif 'Nov' in x: 
        return int(x.replace('Nov', '').strip())
    elif 'Dec' in x:
        return int(x.replace('Dec', '').strip())
    else:
        return int(x)

df['registered_year'] = df['registered_year'].apply(converter_coluna_registered_year)

# Conversão da coluna resale_price
def converter_preco(x):
    if '₹' in x and 'Lakh' in x:
        return float(x.replace('₹', '').replace('Lakh', '').replace(',', '.').strip()) * 100000
    elif '₹' in x and 'Crore' in x:
        return float(x.replace('₹', '').replace('Crore', '').replace(',', '.').strip()) * 10000000
    else:
        return float(x.replace('₹', '').replace(',', '').strip())
    
df['resale_price'] = df['resale_price'].apply(converter_preco)
df.rename(columns={'resale_price': 'resale_price_inr'}, inplace=True) # inr para dizer que está em rupias indianas

# Conversão da coluna max_power
def converter_coluna_max_power(x):
    x = str(x).lower()
    if 'bhp' in x:
        return float(x.split('bhp')[0].replace('bhp', '').strip())
    elif 'hp' in x:
        return float(x.split('hp')[0].replace('hp', '').strip())
    elif 'ps' in x:
        return float(x.split()[0].replace('ps', '').strip()) * 0.98632
    elif 'kw' in x:
        return float(x.split()[0].replace('kw', '').strip()) * 1.34102209
    elif '(' in x:
        return float(x.split('(')[0].strip())
    elif '/' in x:
        return float(x.split('/')[0].strip())
    else:
        return float(x.strip())

df['max_power'] = df['max_power'].apply(converter_coluna_max_power)
df.rename(columns={'max_power': 'max_power_bhp'}, inplace=True)

# Tratando outliers

Q1_resale_price = df['resale_price_inr'].quantile(0.25)
Q3_resale_price = df['resale_price_inr'].quantile(0.75)
IQR_resale_price = Q3_resale_price - Q1_resale_price

Q1_kms_driven = df['kms_driven'].quantile(0.25)
Q3_kms_driven = df['kms_driven'].quantile(0.75)
IQR_kms_driven = Q3_kms_driven - Q1_kms_driven

Q1_max_power_bhp = df['max_power_bhp'].quantile(0.25)
Q3_max_power_bhp = df['max_power_bhp'].quantile(0.75)
IQR_max_power_bhp = Q3_max_power_bhp - Q1_max_power_bhp

df = df[(df['resale_price_inr'] >= Q1_resale_price - 1.5*IQR_resale_price) & (df['resale_price_inr'] <= Q3_resale_price + 1.5*IQR_resale_price) &
        (df['kms_driven'] >= Q1_kms_driven - 1.5*IQR_kms_driven) & (df['kms_driven'] <= Q3_kms_driven + 1.5*IQR_kms_driven) &
        (df['max_power_bhp'] >= Q1_max_power_bhp - 1.5*IQR_max_power_bhp) & (df['max_power_bhp'] <= Q3_max_power_bhp + 1.5*IQR_max_power_bhp)]

# Fim do tratamento

def plot_linear_regression():
    y = df['resale_price_inr']
    colunas_categoricas = ['transmission_type', 'fuel_type', 'owner_type', 'insurance', 'city']
    colunas_numericas = ['registered_year', 'engine_capacity_cc', 'kms_driven', 'max_power_bhp', 'seats', 'mileage_kmpl']
    X = df[colunas_numericas + colunas_categoricas]

    dummies = pd.get_dummies(df[colunas_categoricas], drop_first=True)

    X = pd.concat([X.drop(colunas_categoricas, axis=1), dummies], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    df_pred = pd.DataFrame({
        'Valor Real': y_test,
        'Valor Previsto': y_pred
    })

    fig = px.scatter(
        df_pred,
        x='Valor Real',
        y='Valor Previsto',
        title='Erro de Predição do Modelo',
        labels={'Valor Real': 'Valor Real', 'Valor Previsto': 'Valor Previsto'},
        trendline='ols',
        trendline_color_override='white',
        color_discrete_sequence=['#950004']
    )

    return fig

def plot_random_forest():
    y = df['resale_price_inr']
    colunas_categoricas = ['transmission_type', 'fuel_type', 'owner_type', 'insurance', 'city']
    colunas_numericas = ['registered_year', 'engine_capacity_cc', 'kms_driven', 'max_power_bhp', 'seats', 'mileage_kmpl']
    X = df[colunas_numericas + colunas_categoricas]

    dummies = pd.get_dummies(df[colunas_categoricas], drop_first=True)

    X = pd.concat([X.drop(colunas_categoricas, axis=1), dummies], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    variaveis_importantes = ['max_power_bhp', 'registered_year', 'engine_capacity_cc', 'mileage_kmpl', 'kms_driven', 'transmission_type_Manual', 'city_Bangalore']

    X_train = X_train[variaveis_importantes]
    X_test = X_test[variaveis_importantes]

    rf = RandomForestRegressor(n_estimators=100, random_state=5)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    df_pred = pd.DataFrame({
        'Valor Real': y_test,
        'Valor Previsto': y_pred
    })

    fig = px.scatter(
        df_pred,
        x='Valor Real',
        y='Valor Previsto',
        title='Erro de Predição do Modelo',
        labels={'Valor Real': 'Valor Real', 'Valor Previsto': 'Valor Previsto'},
        trendline='ols',
        trendline_color_override='white',
        color_discrete_sequence=['#950004']
    )

    return fig

def dashboard():
    st.set_page_config(page_title='Dashboard', layout='wide')

    with st.sidebar:
        st.title('🎯 Filtragem de dados')
        st.selectbox('Selecione o modelo aplicado', options=['Linear Regression', 'Random Forest'], key='selected_model')
        st.divider()
        st.write('#### Filtragem para análises do dataset')
        st.select_slider('Selecione o intervalo de anos', options=sorted(df['registered_year'].unique()), value=(sorted(df['registered_year'].unique())[0], sorted(df['registered_year'].unique())[-1]), key='selected_years')
        options_1 = {'fuel_type': 'Tipo de combustível', 'transmission_type': 'Tipo de transmissão', 'city': 'Cidade', 'registered_year': 'Ano de registro', 'owner_type': 'Tipo de dono', 'insurance': 'Seguro'}
        st.selectbox('Selecione por qual coluna você deseja agrupar', options=[valor for valor in list(options_1.values())], placeholder="Escolha um ano", key='selected_label')
        selected_col = [k for k, v in options_1.items() if v == st.session_state.selected_label][0]
        st.divider()
        st.write('#### Filtragem para análises voltadas ao modelo')
        options_2 = {'resale_price_inr': 'Preço de revenda', 'registered_year': 'Ano de registro', 'engine_capacity_cc': 'Capacidade do motor (cc)', 'kms_driven': 'Quilometragem (kms)', 'max_power_bhp': 'Potência máxima (bhp)', 'seats': 'Número de assentos', 'mileage_kmpl': 'Consumo (km/l)'}
        st.selectbox(' ', options=[col for col in list(options_2.values())], placeholder="Escolha um ano", key='selected_label_2', label_visibility='collapsed')
        selected_col_num = [k for k, v in options_2.items() if v == st.session_state.selected_label_2][0]
        st.divider()

    with st.container(border=True):
        st.write('## 📊 Análise para predição dos preços de carro na Índia')
        st.divider()
        col1, col2 = st.columns(2)

        df_filtrado = df[(df['registered_year'] >= st.session_state.selected_years[0]) & (df['registered_year'] <= st.session_state.selected_years[1])] 

        cores_vermelhas_hex = sns.color_palette("Reds", len(df[selected_col].unique())).as_hex()

        # Gráfico 1
        try:
            with col1:
                fig1 = px.bar(
                    df_filtrado.groupby(selected_col)['resale_price_inr'].mean().sort_values(ascending=True).reset_index(),
                    x=selected_col,
                    y='resale_price_inr',
                    color=selected_col,
                    title=f'Diferença da média de preços por {st.session_state.selected_label}',
                    labels={selected_col: st.session_state.selected_label, 'resale_price_inr': "Preço de revenda em Rupias Indianas"},
                    color_discrete_sequence=cores_vermelhas_hex,
                    color_continuous_scale='reds'
                ) 
                st.plotly_chart(fig1)
        except Exception as e:
            st.write(f'Erro ao gerar o gráfico: {e}')     

        # Gráfico 2
        try:
            with col2:
                fig2 = px.line(
                    df_filtrado.groupby('registered_year')['resale_price_inr'].mean().reset_index(),
                    x='registered_year',
                    y='resale_price_inr',
                    title='Preço médio de carros por ano de registro',
                    labels={'registered_year': "Ano de registro", 'resale_price_inr': "Preço de revenda em Rupias Indianas"},
                    color_discrete_sequence=['#950004']
                ) 
                st.plotly_chart(fig2)
        except Exception as e:
            st.write(f'Erro ao gerar o gráfico: {e}')

        if st.session_state.selected_model == 'Linear Regression':
            st.divider()
            st.write(f'## 📈 Análises para aplicar o modelo {st.session_state.selected_model} e visualização do erro de predição')
            st.divider()
        else:
            st.divider()
            st.write(f'## 🌲 Análises para aplicar o modelo {st.session_state.selected_model} e visualização do erro de predição')
            st.divider()

        col1, col2 = st.columns(2)
        # Gráfico 3
        try:
            with col1:
                fig3 = px.histogram(
                    df_filtrado,
                    x=selected_col_num,
                    nbins=20,
                    title=f'Distribuição do dataframe por {st.session_state.selected_label_2}',
                    labels={'count': "Quantidade", selected_col_num: st.session_state.selected_label_2},
                    color_discrete_sequence=['#950004']
                )

                st.plotly_chart(fig3.update_traces(marker_line_color='black', marker_line_width=1))
        except Exception as e:
            st.write(f'Erro ao gerar o gráfico: {e}')

        # Gráfico 4
        try:
            with col2:
                fig4 = px.scatter(
                    df_filtrado,
                    x=selected_col_num,
                    y='resale_price_inr',
                    title=f'Relação entre {st.session_state.selected_label_2} e Preço de Revenda',
                    labels={selected_col_num: st.session_state.selected_label_2, 'resale_price_inr': "Preço de revenda em Rupias Indianas"},
                    color_discrete_sequence=['#950004']
                )

                st.plotly_chart(fig4)
        except Exception as e:
            st.write(f'Erro ao gerar o gráfico: {e}')

        col1, col2, col3 = st.columns([2,6,2])
        # Gráfico 5
        try:
            with col2:
                if st.session_state.selected_model == 'Linear Regression':
                    st.plotly_chart(plot_linear_regression())
                else:
                    st.plotly_chart(plot_random_forest())
        except Exception as e:
            st.write(f'Erro ao gerar o gráfico: {e}')    

if __name__ == '__main__':
    dashboard()