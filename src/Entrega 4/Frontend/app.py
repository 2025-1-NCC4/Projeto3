import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Configuração da página
st.set_page_config(page_title="Indicadores Econômicos", layout="wide", initial_sidebar_state="expanded")

# Título e descrição
st.markdown("# 📈 Indicadores Econômicos")
st.markdown("Este painel interativo apresenta dados históricos e projeções econômicas com base em informações do Banco Central do Brasil.")
st.markdown("---")

# Carregamento dos dados
df = pd.read_excel("dados_mesclados.xlsx")
df['Data'] = pd.to_datetime(df['Data'])

# Seletor de indicadores
st.sidebar.header("Configurações de Visualização")
opcoes = st.sidebar.multiselect(
    "Selecione os indicadores para exibição:",
    ['Salario_Minimo', 'IPCA', 'Endividamento'],
    default=['Salario_Minimo', 'IPCA', 'Endividamento']
)

# Multiplicadores para ajuste visual
mult = {'Salario_Minimo': 1, 'IPCA': 100, 'Endividamento': 10}

# Gráfico interativo dos dados históricos
st.subheader("📊 Evolução Histórica dos Indicadores")
fig = go.Figure()
for indicador in opcoes:
    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df[indicador] * mult[indicador],
        mode='lines',
        name=f'{indicador} (x{mult[indicador]})'
    ))

fig.update_layout(
    xaxis_title='Data',
    yaxis_title='Valores Ajustados',
    template='ggplot2',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Previsão do Salário Mínimo
st.subheader("Previsão do Salário Mínimo (Modelo: Prophet)")
df_prophet = df[['Data', 'Salario_Minimo']].rename(columns={"Data": "ds", "Salario_Minimo": "y"})
modelo = Prophet()
modelo.fit(df_prophet)
future = modelo.make_future_dataframe(periods=36, freq='M')
forecast = modelo.predict(future)

# Gráfico de previsão
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Histórico', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão', line=dict(color='orange')))
fig2.update_layout(
    title="Projeção do Salário Mínimo para os Próximos 3 Anos",
    xaxis_title="Data",
    yaxis_title="Salário Mínimo (R$)",
    template="seaborn"
)
st.plotly_chart(fig2, use_container_width=True)

# Dados brutos
with st.expander("📄 Ver tabela de dados"):
    st.dataframe(df)
