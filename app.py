import streamlit as st
import joblib
import numpy as np

# Carrega o modelo
modelo = joblib.load('modelo_titanic.pkl')

st.title("🛳️ Preditor de Sobrevivência no Titanic")

# Inputs
pclass = st.selectbox("Classe da Passagem (1 = Alta, 3 = Baixa)", [1, 2, 3])
age = st.slider("Idade", 0, 100, 30)
sibsp = st.slider("Nº de irmãos/cônjuges a bordo", 0, 10, 0)
parch = st.slider("Nº de pais/filhos a bordo", 0, 10, 0)
fare = st.number_input("Tarifa paga (em dólares)", value=32.2)
sex_male = st.selectbox("Sexo", ['Feminino', 'Masculino']) == 'Masculino'
embarked_Q = st.checkbox("Embarcou em Queenstown?")
embarked_S = st.checkbox("Embarcou em Southampton?")

# Previsão
input_data = np.array([[pclass, age, sibsp, parch, fare, int(sex_male), int(embarked_Q), int(embarked_S)]])
if st.button("🔮 Prever"):
    resultado = modelo.predict(input_data)
    if resultado[0] == 1:
        st.success("🎉 Sobreviveu!")
    else:
        st.error("💀 Não Sobreviveu.")
