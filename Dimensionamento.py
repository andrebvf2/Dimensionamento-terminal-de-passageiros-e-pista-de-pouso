# DIMENSIONAMENTO PRELIMINAR E PREVISÃO DE DEMANDA
# LEITOR DE INPUT + DADOS IBGE CSV + DADOS ANAC CSV + SERIES TEMPORAIS + TERMINAL E PISTA

from erros import erro, validar_numero, validar_inteiro, validar_positivo, ErroInput
import numpy as np
import pandas as pd
import math
import unicodedata
import re
import os
import logging

# Configuração de log sugerida
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# FUNÇÕES AUXILIARES

def normalizar(txt):
    """Remove acentos, espaços extras, parênteses e converte para minúsculas."""
    if pd.isna(txt): return ""
    # Remove o que estiver dentro de parênteses (comum na base da ANAC: "CIDADE (UF)")
    txt = re.sub(r'\(.*?\)', '', str(txt))
    # Remove códigos numéricos do IBGE no início da string
    txt = re.sub(r'^\d+\s*', '', txt)
    return unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode().lower().strip()


# BLOCOS DO INPUT

BLOCOS = ["POPULACAO", "ALTITUDE", "TEMPERATURA", "DECLIVIDADE", "ENVERGADURA", "DEMANDA_ANUAL", "NIVEL_SERVICO"]

def eh_bloco(linha):
    return linha in BLOCOS


# CARREGAMENTO E CONSOLIDAÇÃO DE DADOS

def carregar_bases_locais():
    """Lê População, PIB (PIB2023.csv) e ANAC, unindo-os de forma resiliente."""
    files = ["ibge_limpo.csv", "PIB2023.csv", "anac.csv"]
    for f in files:
        if not os.path.exists(f):
            raise ErroInput("ARQUIVO", f"Arquivo '{f}' não encontrado na pasta atual.")

    try:
        # 1. Carrega População
        df_pop = pd.read_csv("ibge_limpo.csv", encoding='utf-8')
        df_pop["cidade_norm"] = df_pop["cidade"].apply(normalizar)

        # 2. Carrega PIB (PIB2023.csv)
        # O PIB do SIDRA tem cabeçalho na linha 3
        df_pib_raw = pd.read_csv("PIB2023.csv", sep=None, engine='python', encoding='utf-8-sig', skiprows=3)
        df_pib_raw.columns = df_pib_raw.columns.str.strip().str.lower()
        
        col_cid_pib = [c for c in df_pib_raw.columns if 'mun' in c or 'cid' in c][0]
        col_val_pib = [c for c in df_pib_raw.columns if '2023' in c or 'valor' in c or 'pib' in c][0]
        
        df_pib = pd.DataFrame({
            'cidade_norm': df_pib_raw[col_cid_pib].apply(normalizar),
            'pib_valor': pd.to_numeric(df_pib_raw[col_val_pib], errors='coerce')
        }).dropna()

        # 3. Une População e PIB (Merge)
        df_ibge = pd.merge(df_pop, df_pib, on="cidade_norm", how="left")
        # Se o PIB vier vazio de algum arquivo, preenche-se com 0 para evitar erro no cálculo do score
        df_ibge["pib"] = df_ibge["pib_valor"].fillna(0)

        # 4. Carrega ANAC
        df_anac = pd.read_csv("anac.csv", sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
        df_anac.columns = df_anac.columns.str.strip().str.lower()
        
        col_mun_anac = [c for c in df_anac.columns if 'mun' in c or 'cid' in c][0]
        df_anac = df_anac.rename(columns={col_mun_anac: 'municipio'})
        df_anac["mun_norm"] = df_anac["municipio"].apply(normalizar)

        print(">> Bases locais (População + PIB2023 + ANAC) consolidadas!")
        return df_ibge, df_anac
        
    except Exception as e:
        raise ErroInput("DADOS", f"Erro ao processar as bases: {e}")

def get_dados_ibge(cidade_nome, df_ibge):
    cidade_norm = normalizar(cidade_nome)
    linha = df_ibge[df_ibge["cidade_norm"] == cidade_norm]
    
    if linha.empty:
        # Tenta uma busca parcial caso não encontre exato
        linha = df_ibge[df_ibge["cidade_norm"].str.contains(cidade_norm, na=False)]
        
    if linha.empty:
        raise ErroInput("IBGE", f"Cidade '{cidade_nome}' não encontrada nos arquivos do IBGE.")
        
    populacao = float(linha.iloc[0]["populacao"])
    pib = float(linha.iloc[0]["pib"])
    
    # Prevenção: Se o PIB for zero, usamos uma média regional para não quebrar o score
    if pib <= 0:
        pib = 20000.0 
        
    return populacao, pib

# PREVISÃO DE DEMANDA (ANAC + MMQ)

def selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac):
    """Cruza IBGE e ANAC garantindo que os nomes normalizados batam."""
    cidades_com_aeroporto = df_anac["mun_norm"].unique()
    
    # Filtra no IBGE apenas cidades que existem na base da ANAC
    df_filtrado = df_ibge[df_ibge["cidade_norm"].isin(cidades_com_aeroporto)].copy()
    
    if df_filtrado.empty:
        exemplo_anac = cidades_com_aeroporto[:3]
        exemplo_ibge = df_ibge["cidade_norm"].iloc[:3].tolist()
        raise ErroInput("DADOS", f"Não foi possível cruzar IBGE e ANAC.\nEx ANAC: {exemplo_anac}\nEx IBGE: {exemplo_ibge}")

    # Cálculo de Score Logarítmico
    pop_alvo_log = np.log1p(pop_alvo)
    pib_alvo_log = np.log1p(pib_alvo)

    df_filtrado["score"] = (
        abs(np.log1p(df_filtrado["populacao"]) - pop_alvo_log) / pop_alvo_log +
        abs(np.log1p(df_filtrado["pib"]) - pib_alvo_log) / pib_alvo_log
    )
    
    return df_filtrado.sort_values("score")["cidade_norm"].head(8).tolist()

def obter_series_anac(cidades_norm, df_anac):
    historico = []
    for cidade in cidades_norm:
        df_cidade = df_anac[df_anac["mun_norm"] == cidade]
        if not df_cidade.empty:
            df_cidade = df_cidade.sort_values("ano_mes")
            if len(df_cidade) >= 60:
                serie = df_cidade["passageiros"].tail(60).values
                historico.append(serie.tolist())
    return historico

def prever_demanda_cidade(demandas_60_meses, anos_projecao, ano_base):
    n_meses = 60
    # Lógica de MMQ 
    mm_desc = [sum(demandas_60_meses[i:i+12]) / 12 for i in range(n_meses - 11)]
    mmc = [None] * 6
    for i in range(len(mm_desc) - 1):
        mmc.append((mm_desc[i] + mm_desc[i+1]) / 2)
    mmc.extend([None] * 6)
    
    is_mensal = [demandas_60_meses[i] / mmc[i] if mmc[i] else None for i in range(n_meses)]
    is_medio = []
    for mes in range(12):
        valores_mes = [is_mensal[i] for i in range(mes, n_meses, 12) if is_mensal[i] is not None]
        is_medio.append(sum(valores_mes) / len(valores_mes) if valores_mes else 1.0)
        
    tendencia = [demandas_60_meses[i] / is_medio[i % 12] for i in range(n_meses)]
    soma_x = sum(range(1, n_meses + 1))
    soma_y = sum(tendencia)
    soma_xy = sum((i+1) * tendencia[i] for i in range(n_meses))
    soma_x2 = sum((i+1)**2 for i in range(n_meses))
    
    den = (n_meses * soma_x2 - soma_x**2)
    if den == 0: return {ano: 0 for ano in anos_projecao}
        
    a = (n_meses * soma_xy - soma_x * soma_y) / den
    b = (soma_y - a * soma_x) / n_meses
    
    resultados = {}
    for ano in anos_projecao:
        distancia_anos = ano - ano_base
        mes_fim = distancia_anos * 12
        mes_inicio = mes_fim - 12
        demanda_anual = 0
        for i in range(mes_inicio, mes_fim):
            nova_tendencia = a * (i + 1) + b
            demanda_anual += max(0, nova_tendencia * is_medio[i % 12])
        resultados[ano] = demanda_anual
    return resultados

def calcular_demanda_real(cidade, anos, ano_base):
    df_ibge, df_anac = carregar_bases_locais()
    pop_alvo, pib_alvo = get_dados_ibge(cidade, df_ibge)
    
    print(f">> Perfil: {cidade.title()} | Pop: {pop_alvo:,.0f} | PIB: R$ {pib_alvo:,.2f}")
    
    cidades_similares = selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac)
    print(f">> Cidades Similares ANAC: {', '.join(cidades_similares)}")
    
    series = obter_series_anac(cidades_similares, df_anac)
    
    if len(series) < 3:
        raise ErroInput("DADOS", "Poucas cidades similares encontradas com histórico de 5 anos.")
        
    resultado = {ano: 0 for ano in anos}
    for serie in series:
        prev = prever_demanda_cidade(serie, anos, ano_base)
        for ano in anos:
            resultado[ano] += prev[ano] / len(series)
            
    return resultado

# LEITURA DE INPUT

def ler_populacao(linha):
    v = linha.split()
    if len(v) < 3:
        erro("POPULACAO", "dados incompletos. Forneça: ANO_INICIO INTERVALO NUM_INTERVALOS")
    
    ano_inicio = validar_inteiro(v[0], "ANO_INICIO")
    intervalo = validar_inteiro(v[1], "INTERVALO")
    n = validar_inteiro(v[2], "NUM_INTERVALOS")
    
    anos = [ano_inicio + (intervalo * i) for i in range(1, n+1)]
    return anos, ano_inicio

def ler_arquivo_input(caminho):
    dados = {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = [l.strip() for l in f if l.strip() != ""]
    except FileNotFoundError:
        erro("ARQUIVO", "input.txt não encontrado.")
        
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        if linha == "POPULACAO":
            dados["ANOS_PROJ"], dados["ANO_ZERO"] = ler_populacao(linhas[i+1])
        elif linha == "ALTITUDE":
            dados["ALTITUDE"] = validar_numero(linhas[i+1], "ALTITUDE")
        elif linha == "TEMPERATURA":
            dados["TEMPERATURA"] = validar_numero(linhas[i+1], "TEMPERATURA")
        elif linha == "DECLIVIDADE":
            dados["DECLIVIDADE"] = validar_numero(linhas[i+1], "DECLIVIDADE")
        elif linha == "ENVERGADURA":
            dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")
        elif linha == "DEMANDA_ANUAL":
            valor = linhas[i+1].upper()
            if "CALCULAR" in valor:
                partes = valor.split()
                if len(partes) < 2:
                    erro("DEMANDA_ANUAL", "Cidade não informada após CALCULAR. Ex: CALCULAR Picos")
                
                dados["CIDADE_ALVO"] = " ".join(partes[1:]).title() 
                dados["DEMANDA_ANUAL"] = "CALCULAR"
            else:
                dados["DEMANDA_ANUAL"] = float(valor)
        elif linha == "NIVEL_SERVICO":
            nivel = linhas[i+1].upper()
            if nivel not in ["A", "B", "C"]:
                erro("NIVEL_SERVICO", "use A, B ou C.")
            dados["NIVEL_SERVICO"] = nivel
        i += 1
    return dados


# CÁLCULOS FÍSICOS E ÁREAS

#Calculo da Area dos terminais

def fator_hora_pico(d):
    if d < 100000: return 0.169
    elif d < 500000: return 0.068
    elif d < 1000000: return 0.064
    elif d < 10000000: return 0.027
    else: return 0.024

def calcular_php(d):
    fhp = fator_hora_pico(d)
    php = d * fhp / 100
    return php, fhp

def dimensionar_terminal(php, nivel):
    indices_embarque = {"A": 1.80, "B": 1.50, "C": 1.20}
    indices_pre = {"A": 1.20, "B": 1.00, "C": 0.80}
    area_balcao = {"A": 24.12, "B": 19.20, "C": 15.21}
    percentual_bilhetes = {"A": 0.35, "B": 0.25, "C": 0.15}
    area_bilhete_unit = {"A": 6.48, "B": 5.70, "C": 5.04}
    indices_bagagem = {"A": 1.30, "B": 1.10, "C": 0.80}
    indices_desembarque = {"A": 1.50, "B": 1.20, "C": 1.00}

    balcoes = math.ceil((0.20 * php) / 7.5)
    n_bilhetes = max(1, math.ceil(balcoes * percentual_bilhetes[nivel]))

    areas = {
        "Saguão embarque": php * indices_embarque[nivel],
        "Pré-embarque": php * indices_pre[nivel],
        "Check-in": balcoes * area_balcao[nivel],
        "Bilhetes": n_bilhetes * area_bilhete_unit[nivel],
        "Segurança": math.ceil(php / 180) * 13.5,
        "Triagem e despacho": math.ceil(php / 70) * 20,
        "Restituição bagagem": php * indices_bagagem[nivel],
        "Saguão desembarque": php * indices_desembarque[nivel]
    }
    
    return areas, sum(areas.values()), balcoes, n_bilhetes

#Calculo da pista de pouso do aeroporto 

def obter_L0(e):
    if e < 15: return 800
    elif e < 24: return 1000
    elif e < 36: return 1200
    elif e < 52: return 1800
    elif e < 65: return 2400
    else: return 3000

def calcular_pista(L0, alt, temp, decl):
    CA = (alt/300)*0.07 + 1
    TP = 15 - 0.0066*alt
    CT = 1 + (temp - TP)*0.01
    CD = 1 + decl*0.10
    return L0 * CA * CT * CD

def largura_pista(L, e):
    if L < 800:
        return 18 if e < 24 else 23
    elif L < 1200:
        return 23 if e < 24 else 30
    elif L < 1800:
        return 30 if e < 36 else 45
    else:
        if e < 65: return 45
        elif e <= 80: return 60
    return None

# MAIN

if __name__ == "__main__":
    try:
        dados = ler_arquivo_input("input.txt")
        anos = dados["ANOS_PROJ"]
        ano_zero = dados["ANO_ZERO"]
        nivel = dados["NIVEL_SERVICO"]
        
        L0 = obter_L0(dados["ENVERGADURA"])
        Lf = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["DECLIVIDADE"])
        largura = largura_pista(Lf, dados["ENVERGADURA"])

        print("\n==== INFRAESTRUTURA DA PISTA ====")
        print(f"Comprimento: {Lf:.2f} m")
        print(f"Largura: {largura if largura else 'Não aplicável'} m")

        if dados.get("DEMANDA_ANUAL") == "CALCULAR":
            cidade_alvo = dados["CIDADE_ALVO"]
            print(f"\n>> Iniciando processamento de séries temporais para o horizonte {anos}...")
            demandas_anuais = calcular_demanda_real(cidade_alvo, anos, ano_zero)
        else:
            demandas_anuais = {ano: dados["DEMANDA_ANUAL"] for ano in anos}

        for ano in anos:
            d = demandas_anuais[ano]
            php, fhp = calcular_php(d)
            areas, total, balcoes, n_bilhetes = dimensionar_terminal(php, nivel)
            
            print(f"\n{'='*15} ANO DE PROJETO: {ano} {'='*15}")
            print(f"Demanda Anual Projetada: {d:.0f} pax/ano")
            print(f"FHP: {fhp} | PHP: {php:.2f} pax/hora-pico")
            
            print("\n-- DIMENSIONAMENTO DO TERMINAL --")
            print(f"Balcões de check-in: {balcoes}")
            print(f"Balcões de venda de bilhetes: {n_bilhetes}")
            print("\nÁreas Calculadas:")
            
            for k, v in areas.items():
                print(f"  > {k}: {v:.2f} m²")
                
            print(f"\n>> ÁREA TOTAL DO TERMINAL: {total:.2f} m²")

    except ErroInput as e:
        print("\n*** ERRO DE EXECUÇÃO ***")
        print(e)
    except Exception as e:
        print(f"\n*** ERRO INESPERADO ***\n{e}")
