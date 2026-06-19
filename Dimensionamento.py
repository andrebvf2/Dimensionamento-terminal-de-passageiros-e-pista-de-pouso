# DIMENSIONAMENTO PRELIMINAR E PREVISÃO DE DEMANDA
# MAIN - ORQUESTRADOR

import numpy as np
import pandas as pd
import math
import unicodedata
import re
import os
import logging

# IMPORTANDO OS NOSSOS NOVOS MÓDULOS OTIMIZADOS
from terminal import calcular_php, dimensionar_terminal
from pista import calcular_pista, largura_pista, determinar_configuracao_pista, buscar_ventos_local

# Configuração de log sugerida
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =========================
# FUNÇÕES DE ERRO E FORMATAÇÃO
# =========================
class ErroInput(Exception):
    def __init__(self, bloco, mensagem):
        self.bloco = bloco
        self.mensagem = mensagem
        super().__init__(f"[{bloco}] {mensagem}")

def erro(bloco, mensagem):
    raise ErroInput(bloco, mensagem)

def validar_numero(valor, nome_bloco):
    try:
        return float(valor)
    except ValueError:
        erro(nome_bloco, f"O valor '{valor}' não é um número válido.")

def validar_inteiro(valor, nome_bloco):
    try:
        return int(valor)
    except ValueError:
        erro(nome_bloco, f"O valor '{valor}' não é um número inteiro válido.")

def validar_positivo(valor, nome_bloco):
    num = validar_numero(valor, nome_bloco)
    if num <= 0:
        erro(nome_bloco, f"O valor precisa ser maior que zero. Recebido: {num}")
    return num

def formato_br(valor, casas=2):
    if valor is None: return "N/A"
    texto = f"{valor:,.{casas}f}"
    texto = texto.replace(',', 'X').replace('.', ',').replace('X', '.')
    return texto

def normalizar(txt):
    if pd.isna(txt): return ""
    txt = re.sub(r'\(.*?\)', '', str(txt))
    txt = re.sub(r'^\d+\s*', '', txt)
    return unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode().lower().strip()

# =========================
# BLOCOS DO INPUT E DADOS
# =========================
BLOCOS = ["POPULACAO", "ALTITUDE", "TEMPERATURA", "COTAS_PISTA", "ENVERGADURA", "DEMANDA_ANUAL", "NIVEL_SERVICO", "COMPRIMENTO_BASICO", "MODELO_REGRESSAO"]

def carregar_bases_locais():
    files = ["ibge_limpo.csv", "PIB2023.csv", "anac.csv"]
    for f in files:
        if not os.path.exists(f):
            raise ErroInput("ARQUIVO", f"Arquivo '{f}' não encontrado na pasta atual.")
    try:
        df_pop = pd.read_csv("ibge_limpo.csv", encoding='utf-8')
        df_pop["cidade_norm"] = df_pop["cidade"].apply(normalizar)

        df_pib_raw = pd.read_csv("PIB2023.csv", sep=None, engine='python', encoding='utf-8-sig', skiprows=3)
        df_pib_raw.columns = df_pib_raw.columns.str.strip().str.lower()
        col_cid_pib = [c for c in df_pib_raw.columns if 'mun' in c or 'cid' in c][0]
        col_val_pib = [c for c in df_pib_raw.columns if '2023' in c or 'valor' in c or 'pib' in c][0]
        df_pib = pd.DataFrame({
            'cidade_norm': df_pib_raw[col_cid_pib].apply(normalizar),
            'pib_valor': pd.to_numeric(df_pib_raw[col_val_pib], errors='coerce')
        }).dropna()

        df_ibge = pd.merge(df_pop, df_pib, on="cidade_norm", how="left")
        df_ibge["pib"] = df_ibge["pib_valor"].fillna(0)

        df_anac = pd.read_csv("anac.csv", sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
        df_anac.columns = df_anac.columns.str.strip().str.lower()
        col_mun_anac = [c for c in df_anac.columns if 'mun' in c or 'cid' in c][0]
        df_anac = df_anac.rename(columns={col_mun_anac: 'municipio'})
        df_anac["mun_norm"] = df_anac["municipio"].apply(normalizar)

        print(">> Bases locais consolidadas!")
        return df_ibge, df_anac
    except Exception as e:
        raise ErroInput("DADOS", f"Erro ao processar as bases: {e}")

def get_dados_ibge(cidade_nome, df_ibge):
    cidade_norm = normalizar(cidade_nome)
    linha = df_ibge[df_ibge["cidade_norm"] == cidade_norm]
    if linha.empty: linha = df_ibge[df_ibge["cidade_norm"].str.contains(cidade_norm, na=False)]
    if linha.empty: raise ErroInput("IBGE", f"Cidade '{cidade_nome}' não encontrada nos arquivos do IBGE.")
    populacao = float(linha.iloc[0]["populacao"])
    pib = float(linha.iloc[0]["pib"])
    if pib <= 0: pib = 20000.0 
    return populacao, pib

# =========================
# MODELOS ESTATÍSTICOS E DEMANDA
# =========================
class FuncoesRegressao:
    @staticmethod
    def linear_ancorada(x_arr, y_arr):
        X1 = 1
        Y1_rounded = round(y_arr[0]) if len(y_arr) > 0 else 0
        num = sum((x_arr[i] - X1) * (y_arr[i] - Y1_rounded) for i in range(len(x_arr)))
        den = sum((x_arr[i] - X1)**2 for i in range(len(x_arr)))
        a = num / den if den != 0 else 0
        b = Y1_rounded - a
        return lambda x: a * x + b, f"Y = {a:.4f} * X + {b:.4f}"

    @staticmethod
    def minimos_quadrados(x_arr, y_arr):
        n = len(x_arr)
        sum_x = sum(x_arr)
        sum_y = sum(y_arr)
        sum_xy = sum(x * y for x, y in zip(x_arr, y_arr))
        sum_x2 = sum(x**2 for x in x_arr)
        den = (n * sum_x2 - sum_x**2)
        a = (n * sum_xy - sum_x * sum_y) / den if den != 0 else 0
        b = (sum_y - a * sum_x) / n
        return lambda x: a * x + b, f"Y = {a:.4f} * X + {b:.4f}"

    @staticmethod
    def logaritmica(x_arr, y_arr):
        n = len(x_arr)
        ln_x = [math.log(x) for x in x_arr]
        sum_lnx = sum(ln_x)
        sum_y = sum(y_arr)
        sum_lnx_y = sum(lx * y for lx, y in zip(ln_x, y_arr))
        sum_lnx2 = sum(lx**2 for lx in ln_x)
        den = (n * sum_lnx2 - sum_lnx**2)
        a = (n * sum_lnx_y - sum_lnx * sum_y) / den if den != 0 else 0
        b = (sum_y - a * sum_lnx) / n
        return lambda x: a * math.log(x) + b, f"Y = {a:.4f} * ln(X) + {b:.4f}"

    @staticmethod
    def exponencial(x_arr, y_arr):
        n = len(x_arr)
        y_safe = [y if y > 0 else 1e-5 for y in y_arr]
        ln_y = [math.log(y) for y in y_safe]
        sum_x = sum(x_arr)
        sum_lny = sum(ln_y)
        sum_x_lny = sum(x * ly for x, ly in zip(x_arr, ln_y))
        sum_x2 = sum(x**2 for x in x_arr)
        den = (n * sum_x2 - sum_x**2)
        B = (n * sum_x_lny - sum_x * sum_lny) / den if den != 0 else 0
        A_ln = (sum_lny - B * sum_x) / n
        a = math.exp(A_ln)
        return lambda x: a * math.exp(B * x), f"Y = {a:.4f} * e^({B:.4f} * X)"

def selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac):
    cidades_com_aeroporto = df_anac["mun_norm"].unique()
    df_filtrado = df_ibge[df_ibge["cidade_norm"].isin(cidades_com_aeroporto)].copy()
    if df_filtrado.empty: raise ErroInput("DADOS", "Não foi possível cruzar IBGE e ANAC.")

    pop_alvo_log = np.log1p(pop_alvo)
    pib_alvo_log = np.log1p(pib_alvo)

    df_filtrado["score"] = (
        abs(np.log1p(df_filtrado["populacao"]) - pop_alvo_log) / pop_alvo_log +
        abs(np.log1p(df_filtrado["pib"]) - pib_alvo_log) / pib_alvo_log
    )
    anomalias_conhecidas = ["fernando de noronha", "porto seguro", "gramado", "rio de janeiro", "sao paulo", "belem"]
    df_filtrado = df_filtrado[~df_filtrado["cidade_norm"].isin(anomalias_conhecidas)]
    df_filtrado = df_filtrado[(df_filtrado["populacao"] <= pop_alvo * 10) & (df_filtrado["populacao"] >= pop_alvo * 0.1)]
    return df_filtrado.sort_values("score")["cidade_norm"].head(5).tolist()

def obter_series_anac(cidades_norm, df_anac):
    historico = []
    for cidade in cidades_norm:
        df_cidade = df_anac[df_anac["mun_norm"] == cidade]
        if not df_cidade.empty:
            df_cidade = df_cidade.sort_values("ano_mes")
            if len(df_cidade) >= 60:
                serie = df_cidade["passageiros"].tail(60).values
                historico.append((cidade, serie.tolist()))
    return historico

def prever_demanda_cidade(demandas_60_meses, anos_projecao, ano_base, nome_cidade="", imprimir_demonstracao=False, modelo_id=1):
    n_meses = 60
    mm_desc = [sum(demandas_60_meses[i:i+12]) / 12 for i in range(n_meses - 11)]
    mmc = [None] * 6
    for i in range(len(mm_desc) - 1): mmc.append((mm_desc[i] + mm_desc[i+1]) / 2)
    mmc.extend([None] * 6)
    
    is_mensal = [demandas_60_meses[i] / mmc[i] if mmc[i] else None for i in range(n_meses)]
    is_medio = []
    for mes in range(12):
        valores_mes = [is_mensal[i] for i in range(mes, n_meses, 12) if is_mensal[i] is not None]
        is_medio.append(sum(valores_mes) / len(valores_mes) if len(valores_mes) > 0 else 1.0)
        
    tendencia = [demandas_60_meses[i] / is_medio[i % 12] if is_medio[i % 12] != 0 else 0 for i in range(n_meses)]
    x_arr = list(range(1, n_meses + 1))
    
    if modelo_id == 1: pred_func, eq_str = FuncoesRegressao.linear_ancorada(x_arr, tendencia); nome_modelo = "Regressão Linear Ancorada"
    elif modelo_id == 2: pred_func, eq_str = FuncoesRegressao.minimos_quadrados(x_arr, tendencia); nome_modelo = "Regressão Linear (Mínimos Quadrados)"
    elif modelo_id == 3: pred_func, eq_str = FuncoesRegressao.logaritmica(x_arr, tendencia); nome_modelo = "Regressão Logarítmica (LN)"
    elif modelo_id == 4: pred_func, eq_str = FuncoesRegressao.exponencial(x_arr, tendencia); nome_modelo = "Regressão Exponencial"
    else: raise ErroInput("MODELO_REGRESSAO", "Funcão de regressão não cadastrada.")
    
    if imprimir_demonstracao:
        print("\n" + "#"*70)
        print(f"📊 DEMONSTRAÇÃO DA EQUAÇÃO: {nome_cidade.upper()} | {nome_modelo}")
        print(f">> Equação Calculada: {eq_str} <<\n" + "#"*70 + "\n")
    
    resultados = {}
    for ano in anos_projecao:
        delta_anos = ano - ano_base
        mes_inicio = 60 + (delta_anos - 1) * 12
        mes_fim = mes_inicio + 12
        demanda_anual = 0
        for i in range(mes_inicio, mes_fim):
            nova_tendencia = pred_func(i + 1)
            demanda_anual += max(0, nova_tendencia * is_medio[i % 12])
        resultados[ano] = demanda_anual
    return resultados

def calcular_demanda_real(cidade, anos, ano_base, modelo_id):
    df_ibge, df_anac = carregar_bases_locais()
    pop_alvo, pib_alvo = get_dados_ibge(cidade, df_ibge)
    print(f">> Perfil Alvo: {cidade.title()} | Pop: {formato_br(pop_alvo, 0)} | PIB: R$ {formato_br(pib_alvo, 2)}")
    
    cidades_similares = selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac)
    print(f">> Cidades Similares Selecionadas: {', '.join([c.title() for c in cidades_similares])}")
    
    series_com_nomes = obter_series_anac(cidades_similares, df_anac)
    if len(series_com_nomes) == 0: raise ErroInput("DADOS", "Nenhuma cidade similar possui o histórico de 60 meses.")
        
    resultado = {ano: 0 for ano in anos}
    for indice, (cid_similar, serie) in enumerate(series_com_nomes):
        prev_bruta = prever_demanda_cidade(serie, anos, ano_base, cid_similar, (indice == 0), modelo_id)
        pop_similar, _ = get_dados_ibge(cid_similar, df_ibge)
        for ano in anos:
            resultado[ano] += (prev_bruta[ano] / pop_similar) * pop_alvo / len(series_com_nomes)
    return resultado

# =========================
# LEITURA DE INPUT
# =========================
def ler_populacao(linha):
    v = linha.split()
    if len(v) < 3: erro("POPULACAO", "dados incompletos.")
    ano_inicio = validar_inteiro(v[0], "ANO_INICIO")
    intervalo = validar_inteiro(v[1], "INTERVALO")
    n = validar_inteiro(v[2], "NUM_INTERVALOS")
    return [ano_inicio + (intervalo * i) for i in range(1, n+1)], ano_inicio

def ler_arquivo_input(caminho):
    dados = {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = [l.strip() for l in f if l.strip() != ""]
    except FileNotFoundError: erro("ARQUIVO", "input.txt não encontrado.")
        
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        if linha == "POPULACAO": dados["ANOS_PROJ"], dados["ANO_ZERO"] = ler_populacao(linhas[i+1])
        elif linha == "ALTITUDE": dados["ALTITUDE"] = validar_numero(linhas[i+1], "ALTITUDE")
        elif linha == "TEMPERATURA": dados["TEMPERATURA"] = validar_numero(linhas[i+1], "TEMPERATURA")
        elif linha == "COTAS_PISTA":
            v = linhas[i+1].split()
            if len(v) < 2: erro("COTAS_PISTA", "Forneça dois valores. Ex: 565 550")
            dados["COTA_ALTA"], dados["COTA_BAIXA"] = validar_numero(v[0], "COTA_ALTA"), validar_numero(v[1], "COTA_BAIXA")
        elif linha == "ENVERGADURA": dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")
        elif linha == "COMPRIMENTO_BASICO": dados["COMPRIMENTO_BASICO"] = validar_numero(linhas[i+1], "COMPRIMENTO_BASICO")
        elif linha == "MODELO_REGRESSAO": dados["MODELO_REGRESSAO"] = validar_inteiro(linhas[i+1], "MODELO_REGRESSAO")
        elif linha == "DEMANDA_ANUAL":
            valor = linhas[i+1].upper()
            if "CALCULAR" in valor:
                partes = valor.split()
                if len(partes) < 2: erro("DEMANDA_ANUAL", "Cidade não informada após CALCULAR.")
                dados["CIDADE_ALVO"] = " ".join(partes[1:]).title() 
                dados["DEMANDA_ANUAL"] = "CALCULAR"
            else: dados["DEMANDA_ANUAL"] = float(valor)
        elif linha == "NIVEL_SERVICO":
            nivel = linhas[i+1].upper()
            if nivel not in ["A", "B", "C"]: erro("NIVEL_SERVICO", "use A, B ou C.")
            dados["NIVEL_SERVICO"] = nivel
        i += 1
    return dados

# =========================
# MAIN - ORQUESTRADOR
# =========================
if __name__ == "__main__":
    try:
        dados = ler_arquivo_input("input.txt")
        anos, ano_zero, nivel = dados["ANOS_PROJ"], dados["ANO_ZERO"], dados["NIVEL_SERVICO"]
        modelo_escolhido = dados.get("MODELO_REGRESSAO", 1)
        
        if "COMPRIMENTO_BASICO" not in dados: raise ErroInput("COMPRIMENTO_BASICO", "Falta o bloco no input.txt")
        if "COTA_ALTA" not in dados or "COTA_BAIXA" not in dados: raise ErroInput("COTAS_PISTA", "O bloco não foi encontrado.")
        
        L0 = dados["COMPRIMENTO_BASICO"]
        
        # --- UTILIZANDO O MÓDULO PISTA ---
        Lf, ca, ct, cd, decl_calculada = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["COTA_ALTA"], dados["COTA_BAIXA"])
        largura = largura_pista(L0, dados["ENVERGADURA"])

        print("\n==== INFRAESTRUTURA DA PISTA ====")
        print(f"Comprimento Básico (L0): {formato_br(L0)} m")
        print(f"Cotas Topográficas (Alta/Baixa): {formato_br(dados['COTA_ALTA'])}m | {formato_br(dados['COTA_BAIXA'])}m")
        print(f"Declividade Calculada: {formato_br(decl_calculada)}%")
        print(f"Fator CA: {formato_br(ca, 4)} | Fator CT: {formato_br(ct, 4)} | Fator CD: {formato_br(cd, 4)}")
        print(f"Comprimento Corrigido (Lf): {formato_br(Lf)} m")
        print(f"Largura da Pista: {formato_br(largura, 0) if largura else 'Fora das especificações'} m")

        if dados["ENVERGADURA"] < 24: limite_vento = 10.5
        elif dados["ENVERGADURA"] < 36: limite_vento = 13.0
        else: limite_vento = 20.0

        # --- UTILIZANDO O MÓDULO PISTA PARA VENTOS ---
        df_ventos_local = buscar_ventos_local("historico_ventos.csv")
        pista_ideal, cobertura, secundaria = determinar_configuracao_pista(df_ventos_local, limite_vento)
        
        print("\n==== LOCAÇÃO E GEOMETRIA DA PISTA ====")
        print(f"Orientação Magnética: Cabeceiras {pista_ideal}")
        print(f"Cobertura de Vento Calculada: {formato_br(cobertura)}% do tempo operável")

        if secundaria: print(">>> RECOMENDAÇÃO: O projeto exigirá pistas transversais (cobertura < 95%).")
        else: print(">>> CONFIGURAÇÃO: Pista única atende aos requisitos (> 95%).")

        # --- DEMANDA E TERMINAL ---
        if dados.get("DEMANDA_ANUAL") == "CALCULAR":
            print(f"\n>> Iniciando processamento de séries temporais para o horizonte {anos}...")
            demandas_anuais = calcular_demanda_real(dados["CIDADE_ALVO"], anos, ano_zero, modelo_escolhido)
        else:
            demandas_anuais = {ano: dados["DEMANDA_ANUAL"] for ano in anos}

        for ano in anos:
            d = demandas_anuais[ano]
            
            # --- UTILIZANDO O MÓDULO TERMINAL ---
            php, fhp = calcular_php(d)
            areas, total, balcoes, n_bilhetes = dimensionar_terminal(php, nivel)
            
            print(f"\n{'='*15} ANO DE PROJETO: {ano} {'='*15}")
            print(f"Demanda Anual Projetada: {formato_br(d, 0)} pax/ano")
            print(f"PHP: {formato_br(php)} pax/hora-pico")
            
            print(f"\n-- DIMENSIONAMENTO DO TERMINAL (NÍVEL {nivel}) --")
            for k, v in areas.items(): print(f"  > {k:<25}: {formato_br(v)} m²")
            print(f"\n>> ÁREA TOTAL: {formato_br(total)} m²")

    except ErroInput as e: print(f"\n*** ERRO DE EXECUÇÃO ***\n{e.mensagem}")
    except Exception as e: print(f"\n*** ERRO INESPERADO ***\n{e}")
