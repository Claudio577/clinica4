# ... todas as importações и funções anteriores ficam aqui ...

def prever_melhorado(anamnese, modelos, le_mob, le_app, palavras_chave_completo, palavras_chave_graves, features, features_eutanasia):
    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = modelos
    texto_norm = normalizar_texto(anamnese)

    # Debug: veja o texto normalizado
    print("TEXTO NORMALIZADO:", texto_norm)

    doencas_detectadas = [p for p in palavras_chave_completo if p in texto_norm]
    doencas_graves_detectadas = [p for p in palavras_chave_graves if p in texto_norm]

    # Debug: veja as listas e se foram encontradas
    print("PALAVRAS CHAVE COMPLETO:", palavras_chave_completo[:10], "...(total:", len(palavras_chave_completo), ")")
    print("DETECÇÃO:", doencas_detectadas)
    print("GRAVES:", doencas_graves_detectadas)

    # O restante da função segue normalmente...
