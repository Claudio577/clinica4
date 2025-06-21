import streamlit as st
from modelo import (
    carregar_dados,
    treinar_modelos,
    prever_melhorado
)

st.set_page_config(page_title="Previsão Clínica Veterinária", page_icon="🐶")

st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese do paciente para prever os cuidados clínicos e analisar doenças mencionadas.")

# ===== Carregar e treinar =====
try:
    df, df_doencas_graves, palavras_chave_total, palavras_chave_graves = carregar_dados()

    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']

    modelos = treinar_modelos(df, features, features_eutanasia, df_doencas_graves)
    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, le_mob, le_app = modelos
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

# ===== Interface de entrada =====
texto = st.text_area("✍️ Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if not texto.strip():
        st.warning("Digite a anamnese primeiro.")
    else:
        resultado = prever_melhorado(
            anamnese=texto,
            modelos=(modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias),
            le_mob=le_mob,
            le_app=le_app,
            palavras_chave_completo=palavras_chave_total,
            palavras_chave_graves=palavras_chave_graves,
            features=features,
            features_eutanasia=features_eutanasia
        )

        st.subheader("📋 Resultado da Análise")
        st.write(f"**Alta:** {resultado['Alta']}")
        st.write(f"**Internar:** {resultado['Internar']}")
        st.write(f"**Dias Internado:** {resultado['Dias Internado']}")
        st.write(f"**Chance de Eutanásia (%):** {resultado['Chance de Eutanásia (%)']}%")

        st.markdown("**🧬 Doenças Detectadas:**")
        if isinstance(resultado['Doenças Detectadas'], list):
            st.write(", ".join(resultado['Doenças Detectadas']))
        else:
            st.write(resultado['Doenças Detectadas'])

        st.markdown("**🚨 Doenças Graves Detectadas:**")
        if isinstance(resultado['Graves Detectadas'], list):
            st.write(", ".join(resultado['Graves Detectadas']))
        else:
            st.write(resultado['Graves Detectadas'])

