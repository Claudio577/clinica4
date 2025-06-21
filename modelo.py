import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extrair_variavel(padrao, texto, tipo=float, valor_padrao=None):
    match = re.search(padrao, texto)
    if match:
        try:
            return tipo(match.group(1).replace(',', '.'))
        except:
            return valor_padrao
    return valor_padrao

def carregar_dados():
    df = pd.read_csv("data/Casos_Cl_nicos_Simulados.csv")
    df_graves = pd.read_csv("data/doencas_caninas_eutanasia_expandidas.csv")
    df_top = pd.read_csv("data/top100_doencas_caninas.csv")

    palavras_chave_total = [
        normalizar_texto(d) for d in df_top['Doença'].dropna().unique()
    ]
    palavras_chave_graves = [
        normalizar_texto(d) for d in df_graves['Doença'].dropna().unique()
    ]

    return df, df_graves, palavras_chave_total, palavras_chave_graves

def treinar_modelos(df, features, features_eutanasia, df_doencas):
    le_mob = LabelEncoder()
    le_app = LabelEncoder()

    df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
    df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

    palavras_chave = [
        normalizar_texto(d) for d in df_doencas['Doença'].dropna().unique()
    ]

    df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
        lambda d: int(any(p in normalizar_texto(d) for p in palavras_chave))
    )

    X_eutanasia = df[features_eutanasia]
    y_eutanasia = df['Eutanasia']

    X_train, _, y_train, _ = train_test_split(X_eutanasia, y_eutanasia, test_size=0.2, stratify=y_eutanasia, random_state=42)
    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train, y_train)

    modelo_alta = RandomForestClassifier().fit(df[features], df['Alta'])
    modelo_internar = RandomForestClassifier(class_weight='balanced', random_state=42).fit(df[features], df['Internar'])
    modelo_dias = RandomForestRegressor().fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, le_mob, le_app

def prever_melhorado(anamnese, modelos, le_mob, le_app, palavras_chave_completo, palavras_chave_graves, features, features_eutanasia):
    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = modelos
    texto_norm = normalizar_texto(anamnese)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:[.,]\d+)?)\s*(?:graus|c|celsius|ºc)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else 5

    if any(p in texto_norm for p in ["dor intensa", "dor severa", "dor forte"]):
        dor = 10
    elif "dor moderada" in texto_norm:
        dor = 5
    elif any(p in texto_norm for p in ["sem dor", "ausencia de dor"]):
        dor = 0
    else:
        dor = 4

    if any(p in texto_norm for p in ["sem apetite", "nao come", "perda de apetite"]):
        apetite = le_app.transform(["nenhum"])[0] if "nenhum" in le_app.classes_ else 0
    elif "baixo apetite" in texto_norm:
        apetite = le_app.transform(["baixo"])[0] if "baixo" in le_app.classes_ else 0
    else:
        apetite = le_app.transform(["normal"])[0] if "normal" in le_app.classes_ else 0

    if any(p in texto_norm for p in ["sem andar", "nao anda", "imovel"]):
        mobilidade = le_mob.transform(["sem andar"])[0] if "sem andar" in le_mob.classes_ else 0
    elif "mobilidade limitada" in texto_norm or "dificuldade" in texto_norm:
        mobilidade = le_mob.transform(["limitada"])[0] if "limitada" in le_mob.classes_ else 0
    else:
        mobilidade = le_mob.transform(["normal"])[0] if "normal" in le_mob.classes_ else 0

    doencas_detectadas = [p for p in palavras_chave_completo if p in texto_norm]
    graves_detectadas = [p for p in palavras_chave_graves if p in texto_norm]
    tem_doenca_letal = int(len(graves_detectadas) > 0)

    entrada = pd.DataFrame([[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]], columns=features_eutanasia)

    prob_eutanasia = modelo_eutanasia.predict_proba(entrada)[0][1]
    prob_internar = modelo_internar.predict_proba(entrada[features])[0][1]

    sintomas_criticos = [
        "estado irreversivel", "prostracao extrema", "nao levanta", "sem reacao",
        "febre muito alta", "respiracao ofegante", "apatia profunda", "mobilidade inexistente",
        "vomitos frequentes", "sem comer", "vocalizando dor"
    ]
    tem_sintomas_terminais = any(p in texto_norm for p in sintomas_criticos)

    if tem_doenca_letal:
        peso_extra_eutanasia = 0.4
    elif tem_sintomas_terminais:
        peso_extra_eutanasia = 0.6
    else:
        peso_extra_eutanasia = 0.0

    prob_eutanasia = min(prob_eutanasia + peso_extra_eutanasia, 1.0)

    alta = modelo_alta.predict(entrada[features])[0]
    internar = 1 if prob_internar > 0.4 else 0
    dias_prev = modelo_dias.predict(entrada[features])[0]

    if tem_sintomas_terminais or prob_eutanasia > 0.5:
        dias_prev += 2

    dias = int(round(dias_prev))
    if internar == 1 and dias < 1:
        dias = 2
    elif internar == 0:
        dias = 0

    eutanasia_chance = round(prob_eutanasia * 100, 1)

    if internar == 0 and eutanasia_chance < 5:
        alta = 1

    return {
        "Alta": "Sim" if alta == 1 else "Não",
        "Internar": "Sim" if internar == 1 else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": eutanasia_chance,
        "Doenças Detectadas": doencas_detectadas if doencas_detectadas else "Nenhuma identificada",
        "Graves Detectadas": graves_detectadas if graves_detectadas else "Nenhuma grave"
    }
