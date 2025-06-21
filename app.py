import streamlit as st
from modelo import (
    carregar_dados,
    treinar_modelos,
    prever_melhorado
)

st.set_page_config(page_title="Previsão Clínica Veterinária", page_icon="🐶")

st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese do paciente ou use um exemplo abaixo para prever o quadro clínico.")

# ===== Carregar e treinar modelos =====
try:
    df, df_doencas_graves, palavras_chave_total, palavras_chave_graves = carregar_dados()

    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']

    modelos = treinar_modelos(df, features, features_eutanasia, df_doencas_graves)
    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, le_mob, le_app = modelos
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

# ===== Anamnese de exemplo =====
st.markdown("### 📌 Exemplos de Anamnese")

exemplos = {
    "✅ Alta": "Cão apresenta leve prostração e apetite normal. Mobilidade preservada. Sem febre. Temperatura corporal de 38,5 °C. Peso 12 kg, idade 5 anos. Sem histórico clínico relevante.",
    "🟡 Doença tratável": "Paciente canino apresenta vômitos intermitentes, febre moderada (39,2 °C) e leve desidratação. Apetite reduzido e mobilidade um pouco limitada. Peso 15 kg, 8 anos. Diagnóstico prévio de giardíase e doença do carrapato, mas responde bem ao tratamento.",
    "🔴 Doença grave com risco de eutanásia": "Cão idoso com linfoma em estágio avançado, apático, prostração intensa, sem mobilidade, vocaliza dor ao toque abdominal. Sem apetite há 3 dias, temperatura elevada (40,1 °C), peso 20 kg, 13 anos. Histórico de anemia hemolítica autoimune. Não responde a estímulos e apresenta quadro clínico irreversível."
}

col1, col2, col3 = st.columns(3)
if col1.button("✅ Alta"):
    st.session_state.anamnese_texto = exemplos["✅ Alta"]
if col2.button("🟡 Doença tratável"):
    st.session_state.anamnese_texto = exemplos["🟡 Doença tratável"]
if col3.button("🔴 Doença grave"):
    st.session_state.anamnese_texto = exemplos["🔴 Doença grave com risco de eutanásia"]

# ===== Entrada da anamnese =====
st.markdown("### ✍️ Digite ou edite a anamnese abaixo:")

if "anamnese_texto" not in st.session_state:
    st.session_state.anamnese_texto = ""

texto = st.text_area("Anamnese do paciente:", value=st.session_state.anamnese_texto, height=200)

# ===== Botão de análise =====
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

# ===== Botão para nova análise =====
if st.button("🆕 Analisar nova anamnese"):
    st.session_state.anamnese_texto = ""
    st.experimental_rerun()

