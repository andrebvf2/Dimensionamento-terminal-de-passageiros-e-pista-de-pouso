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

# --- NOVAS BIBLIOTECAS PARA A API ---
import requests
import datetime

# Configuração de log sugerida
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# FUNÇÕES AUXILIARES

def normalizar(txt):
    """Remove acentos, espaços extras, parênteses e converte para minúsculas."""
    if pd.isna(txt): return ""
    txt = re.sub(r'\(.*?\)', '', str(txt))
    txt = re.sub(r'^\d+\s*', '', txt)
    return unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode().lower().strip()


# BLOCOS DO INPUT

BLOCOS = ["POPULACAO", "ALTITUDE", "TEMPERATURA", "COTAS_PISTA", "ENVERGADURA", "DEMANDA_ANUAL", "NIVEL_SERVICO", "COMPRIMENTO_BASICO", "MODELO_REGRESSAO"]

def eh_bloco(linha):
    return linha in BLOCOS

# CARREGAMENTO E CONSOLIDAÇÃO DE DADOS


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

        print(">> Bases locais (População + PIB2023 + ANAC) consolidadas!")
        return df_ibge, df_anac
        
    except Exception as e:
        raise ErroInput("DADOS", f"Erro ao processar as bases: {e}")

def get_dados_ibge(cidade_nome, df_ibge):
    cidade_norm = normalizar(cidade_nome)
    linha = df_ibge[df_ibge["cidade_norm"] == cidade_norm]
    
    if linha.empty:
        linha = df_ibge[df_ibge["cidade_norm"].str.contains(cidade_norm, na=False)]
        
    if linha.empty:
        raise ErroInput("IBGE", f"Cidade '{cidade_nome}' não encontrada nos arquivos do IBGE.")
        
    populacao = float(linha.iloc[0]["populacao"])
    pib = float(linha.iloc[0]["pib"])
    
    if pib <= 0:
        pib = 20000.0 
        
    return populacao, pib

def obter_estacao_da_cidade(cidade_alvo):
    """Lê o dicionário de estações lidando com problemas de codificação comuns do Brasil."""
    try:
        # Tenta ler no padrão mundial primeiro, se der erro, usa o padrão latino (Windows)
        try:
            df_dic = pd.read_csv("dicionario_estacoes.csv", encoding="utf-8")
        except UnicodeDecodeError:
            df_dic = pd.read_csv("dicionario_estacoes.csv", encoding="latin-1")
            
        df_dic.columns = df_dic.columns.str.lower()
        
        col_nome = [c for c in df_dic.columns if 'estacao' in c or 'nome' in c][0]
        col_id = [c for c in df_dic.columns if 'id_estacao' in c][0]
        
        df_dic['nome_norm'] = df_dic[col_nome].astype(str).apply(normalizar)
        cidade_norm = normalizar(cidade_alvo)
        
        match = df_dic[df_dic['nome_norm'].str.contains(cidade_norm)]
        if not match.empty:
            return match.iloc[0][col_id]
            
    except Exception as e:
        print(f"\n[!] Erro na leitura do dicionário de estações: {e}")
        print("DICA: Verifique se o arquivo baixado não está compactado (.zip/.gz). Extraia o arquivo real antes de usar!")
    return None

# CLASSE DE MODELOS ESTATÍSTICOS

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

# PREVISÃO DE DEMANDA

def selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac):
    cidades_com_aeroporto = df_anac["mun_norm"].unique()
    df_filtrado = df_ibge[df_ibge["cidade_norm"].isin(cidades_com_aeroporto)].copy()
    
    if df_filtrado.empty:
        raise ErroInput("DADOS", "Não foi possível cruzar IBGE e ANAC.")

    pop_alvo_log = np.log1p(pop_alvo)
    pib_alvo_log = np.log1p(pib_alvo)

    df_filtrado["score"] = (
        abs(np.log1p(df_filtrado["populacao"]) - pop_alvo_log) / pop_alvo_log +
        abs(np.log1p(df_filtrado["pib"]) - pib_alvo_log) / pib_alvo_log
    )
    
    anomalias_conhecidas = ["fernando de noronha", "porto seguro", "gramado", "rio de janeiro", "sao paulo", "belem"]
    df_filtrado = df_filtrado[~df_filtrado["cidade_norm"].isin(anomalias_conhecidas)]
    
    df_filtrado = df_filtrado[
        (df_filtrado["populacao"] <= pop_alvo * 10) & 
        (df_filtrado["populacao"] >= pop_alvo * 0.1)
    ]
    
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
    for i in range(len(mm_desc) - 1):
        mmc.append((mm_desc[i] + mm_desc[i+1]) / 2)
    mmc.extend([None] * 6)
    
    is_mensal = [demandas_60_meses[i] / mmc[i] if mmc[i] else None for i in range(n_meses)]
    
    is_medio = []
    for mes in range(12):
        valores_mes = [is_mensal[i] for i in range(mes, n_meses, 12) if is_mensal[i] is not None]
        is_medio.append(sum(valores_mes) / len(valores_mes) if len(valores_mes) > 0 else 1.0)
        
    tendencia = [
        demandas_60_meses[i] / is_medio[i % 12] if is_medio[i % 12] != 0 else 0 
        for i in range(n_meses)
    ]
    
    x_arr = list(range(1, n_meses + 1))
    
    if modelo_id == 1:
        pred_func, eq_str = FuncoesRegressao.linear_ancorada(x_arr, tendencia)
        nome_modelo = "Regressão Linear Ancorada"
    elif modelo_id == 2:
        pred_func, eq_str = FuncoesRegressao.minimos_quadrados(x_arr, tendencia)
        nome_modelo = "Regressão Linear (Mínimos Quadrados)"
    elif modelo_id == 3:
        pred_func, eq_str = FuncoesRegressao.logaritmica(x_arr, tendencia)
        nome_modelo = "Regressão Logarítmica (LN)"
    elif modelo_id == 4:
        pred_func, eq_str = FuncoesRegressao.exponencial(x_arr, tendencia)
        nome_modelo = "Regressão Exponencial"
    else:
        raise ErroInput("MODELO_REGRESSAO", "Funcão de regressão não cadastrada. Cadastrar na classe FuncoesRegressao")
    
    if imprimir_demonstracao:
        print("\n" + "#"*70)
        print(f"📊 DEMONSTRAÇÃO DA EQUAÇÃO: {nome_cidade.upper()} | {nome_modelo}")
        print("#"*70)
        print(f">> Equação Calculada: {eq_str} <<")
        print("#"*70 + "\n")
    
    resultados = {}
    for ano in anos_projecao:
        delta_anos = ano - ano_base
        mes_inicio = 60 + (delta_anos - 1) * 12
        mes_fim = mes_inicio + 12
        
        demanda_anual = 0
        for i in range(mes_inicio, mes_fim):
            mes_idx = i % 12
            x_atual = i + 1
            nova_tendencia = pred_func(x_atual)
            demanda_mes = max(0, nova_tendencia * is_medio[mes_idx])
            demanda_anual += demanda_mes
            
        resultados[ano] = demanda_anual
    return resultados

def calcular_demanda_real(cidade, anos, ano_base, modelo_id):
    df_ibge, df_anac = carregar_bases_locais()
    pop_alvo, pib_alvo = get_dados_ibge(cidade, df_ibge)
    
    print(f">> Perfil Alvo: {cidade.title()} | Pop: {pop_alvo:,.0f} | PIB: R$ {pib_alvo:,.2f}")
    
    cidades_similares = selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac)
    print(f">> Cidades Similares Selecionadas: {', '.join([c.title() for c in cidades_similares])}")
    
    series_com_nomes = obter_series_anac(cidades_similares, df_anac)
    
    if len(series_com_nomes) == 0:
        raise ErroInput("DADOS", "Nenhuma cidade similar possui o histórico contínuo de 60 meses necessário para a regressão matemática.")
        
    resultado = {ano: 0 for ano in anos}
    
    for indice, (cid_similar, serie) in enumerate(series_com_nomes):
        deve_imprimir = (indice == 0) 
        prev_bruta = prever_demanda_cidade(serie, anos, ano_base, cid_similar, deve_imprimir, modelo_id)
        
        pop_similar, _ = get_dados_ibge(cid_similar, df_ibge)
        
        for ano in anos:
            taxa_per_capita = prev_bruta[ano] / pop_similar
            demanda_cidade_alvo = taxa_per_capita * pop_alvo
            resultado[ano] += demanda_cidade_alvo / len(series_com_nomes)
            
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
            
        elif linha == "COTAS_PISTA":
            v = linhas[i+1].split()
            if len(v) < 2:
                erro("COTAS_PISTA", "Você precisa fornecer dois valores numéricos. Ex: 565 550")
            dados["COTA_ALTA"] = validar_numero(v[0], "COTA_ALTA")
            dados["COTA_BAIXA"] = validar_numero(v[1], "COTA_BAIXA")
            
        elif linha == "ENVERGADURA":
            dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")
        elif linha == "COMPRIMENTO_BASICO":
            dados["COMPRIMENTO_BASICO"] = validar_numero(linhas[i+1], "COMPRIMENTO_BASICO")
            
        elif linha == "MODELO_REGRESSAO":
            dados["MODELO_REGRESSAO"] = validar_inteiro(linhas[i+1], "MODELO_REGRESSAO")
            
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

def calcular_pista(L0, alt, temp, cota_alta, cota_baixa):
    diferenca_cotas = abs(cota_alta - cota_baixa)
    declividade = (diferenca_cotas / L0) * 100
    
    CA = (alt / 300) * 0.07 + 1.00
    TP = 15 - 0.0066 * alt
    CT = 1.00 + (temp - TP) * 0.01
    CD = 1.00 + (declividade * 0.10)
    
    L_corrigido = L0 * CA * CT * CD
    return L_corrigido, CA, CT, CD, declividade

def largura_pista(L0, e):
    if e < 15: letra = 'A'
    elif e < 24: letra = 'B'
    elif e < 36: letra = 'C'
    elif e < 52: letra = 'D'
    elif e < 65: letra = 'E'
    elif e < 80: letra = 'F'
    else: return None  

    if L0 < 800:
        if letra in ['A', 'B']: return 18
        elif letra == 'C': return 23
        else: return None
    elif L0 < 1200:
        if letra in ['A', 'B']: return 23
        elif letra == 'C': return 30
        else: return None
    elif L0 < 1800:
        if letra in ['A', 'B', 'C']: return 30
        elif letra == 'D': return 45
        else: return None
    else:
        if letra in ['A', 'B', 'C', 'D', 'E']: return 45
        elif letra == 'F': return 60
    return None

def calcular_vento_cruzado(vel_vento, dir_vento, rumo_pista):
    angulo = abs(dir_vento - rumo_pista)
    if angulo > 180:
        angulo = 360 - angulo
    vento_cruzado = vel_vento * math.sin(math.radians(angulo))
    return abs(vento_cruzado)

def determinar_configuracao_pista(df_ventos, limite_vc=15):
    melhor_rumo = None
    melhor_cobertura = 0
    for rumo in range(10, 181, 10):
        usabilidade = 0
        for _, row in df_ventos.iterrows():
            vc = calcular_vento_cruzado(row['velocidade'], row['direcao'], rumo)
            if vc <= limite_vc:
                usabilidade += 1
        cobertura = (usabilidade / len(df_ventos)) * 100
        if cobertura > melhor_cobertura:
            melhor_cobertura = cobertura
            melhor_rumo = rumo
    precisa_secundaria = melhor_cobertura < 95.0
    pista_ida = int(melhor_rumo / 10)
    pista_volta = pista_ida + 18
    nome_pista = f"{pista_ida:02d}/{pista_volta:02d}"
    return nome_pista, melhor_cobertura, precisa_secundaria

# --- INTEGRAÇÃO COM API DO INMET ---
def buscar_ventos_inmet_api(codigo_estacao, anos_historico=5):
    """
    Busca o histórico de ventos do INMET contornando o firewall governamental 
    com headers de navegador (User-Agent).
    """
    print(f"\n>> Conectando à API do INMET para a estação {codigo_estacao}...")
    ano_atual = datetime.datetime.now().year
    df_ventos_total = pd.DataFrame()
    
    #  Python se passa por um navegador Chrome padrão
    cabecalhos = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    # Loop de requisições anuais para não engasgar o servidor
    for ano in range(ano_atual - anos_historico, ano_atual):
        data_inicio = f"{ano}-01-01"
        data_fim = f"{ano}-12-31"
        url = f"https://apitempo.inmet.gov.br/estacao/{data_inicio}/{data_fim}/{codigo_estacao}"
        
        try:
            # Enviamos a requisição com o cabeçalho camuflado e aguardamos 40s
            resposta = requests.get(url, headers=cabecalhos, timeout=40)
            resposta.raise_for_status()
            
            # Tenta converter a resposta para JSON
            try:
                dados_json = resposta.json()
            except ValueError:
                print(f"  - {ano}: O servidor bloqueou a requisição ou retornou página HTML.")
                continue
                
            # Verifica se o JSON retornou vazio ou com aviso de falta de dados
            if not dados_json or "mensagem" in dados_json:
                print(f"  - {ano}: Sem dados meteorológicos registrados nesta estação.")
                continue
                
            # Converte para DataFrame e extrai apenas a cinemática dos ventos
            df_ano = pd.DataFrame(dados_json)
            df_temp = pd.DataFrame()
            df_temp['direcao'] = pd.to_numeric(df_ano.get('VEN_DIR'), errors='coerce')
            df_temp['velocidade'] = pd.to_numeric(df_ano.get('VEN_VEL'), errors='coerce')
            df_temp = df_temp.dropna()
            
            # Concatena os dados limpos ao histórico geral
            df_ventos_total = pd.concat([df_ventos_total, df_temp])
            print(f"  - {ano}: {len(df_temp)} registros de vento importados com sucesso.")
            
        except requests.exceptions.RequestException:
            print(f"  - {ano}: Conexão com o INMET interrompida ou lenta demais.")
            continue
            
    if df_ventos_total.empty:
        return None
        
    print(f">> Total consolidado: {len(df_ventos_total)} registros prontos para cálculo.")
    return df_ventos_total
        


# MAIN

if __name__ == "__main__":
    try:
        dados = ler_arquivo_input("input.txt")
        anos = dados["ANOS_PROJ"]
        ano_zero = dados["ANO_ZERO"]
        nivel = dados["NIVEL_SERVICO"]
        
        modelo_escolhido = dados.get("MODELO_REGRESSAO", 1)
        
        if "COMPRIMENTO_BASICO" not in dados:
            raise ErroInput("COMPRIMENTO_BASICO", "Falta o bloco COMPRIMENTO_BASICO no input.txt")
        
        if "COTA_ALTA" not in dados or "COTA_BAIXA" not in dados:
            raise ErroInput("COTAS_PISTA", "O bloco COTAS_PISTA não foi encontrado ou está formatado errado.")
        
        L0 = dados["COMPRIMENTO_BASICO"]
        
        Lf, ca, ct, cd, decl_calculada = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["COTA_ALTA"], dados["COTA_BAIXA"])
        largura = largura_pista(L0, dados["ENVERGADURA"])

        print("\n==== INFRAESTRUTURA DA PISTA ====")
        print(f"Comprimento Básico (L0): {L0} m")
        print(f"Cotas Topográficas (Alta/Baixa): {dados['COTA_ALTA']}m | {dados['COTA_BAIXA']}m")
        print(f"Declividade Calculada: {decl_calculada:.2f}%")
        print(f"Fator CA: {ca:.4f} | Fator CT: {ct:.4f} | Fator CD: {cd:.4f}")
        print(f"Comprimento Corrigido (Lf): {Lf:.2f} m")
        print(f"Largura da Pista: {largura if largura else 'Fora das especificações'} m")

        # --- ANÁLISE DE VENTOS VIA API ---
        if dados["ENVERGADURA"] < 24: limite_vento = 10.5
        elif dados["ENVERGADURA"] < 36: limite_vento = 13.0
        else: limite_vento = 20.0

        codigo_estacao = None
        if dados.get("DEMANDA_ANUAL") == "CALCULAR":
            cidade_alvo = dados["CIDADE_ALVO"]
            codigo_estacao = obter_estacao_da_cidade(cidade_alvo)
            
        # Se a cidade não estiver no dicionário, usa Teresina como backup de segurança
        if not codigo_estacao:
            print("\n[!] Cidade não encontrada no dicionário de estações ou nome inválido.")
            print(">> Aplicando a estação meteorológica A312 (Teresina) como Padrão/Teste.")
            codigo_estacao = "A312"

        # Conecta na internet e baixa os ventos!
        df_ventos_api = buscar_ventos_inmet_api(codigo_estacao, anos_historico=5)

        # Se a internet cair ou o INMET estiver fora do ar, cria dados simulados para não travar o projeto
        if df_ventos_api is None or df_ventos_api.empty:
            print("\n[!] Gerando ventos sintéticos de segurança devido a falha na API...")
            df_ventos_api = pd.DataFrame({
                'direcao': np.random.randint(0, 360, 1000),
                'velocidade': np.random.uniform(2, 25, 1000)
            })

        pista_ideal, cobertura, secundaria = determinar_configuracao_pista(df_ventos_api, limite_vento)
        
        print("\n==== LOCAÇÃO E GEOMETRIA DA PISTA ====")
        print(f"Orientação Magnética: Cabeceiras {pista_ideal}")
        print(f"Cobertura de Vento Calculada: {cobertura:.2f}% do tempo operável")

        if secundaria: print(">>> RECOMENDAÇÃO: O projeto exigirá pistas transversais (cobertura < 95%).")
        else: print(">>> CONFIGURAÇÃO: Pista única atende aos requisitos (> 95%).")

        # --- DEMANDA ---
        if dados.get("DEMANDA_ANUAL") == "CALCULAR":
            cidade_alvo = dados["CIDADE_ALVO"]
            print(f"\n>> Iniciando processamento de séries temporais para o horizonte {anos}...")
            demandas_anuais = calcular_demanda_real(cidade_alvo, anos, ano_zero, modelo_escolhido)
        else:
            demandas_anuais = {ano: dados["DEMANDA_ANUAL"] for ano in anos}

        for ano in anos:
            d = demandas_anuais[ano]
            php, fhp = calcular_php(d)
            areas, total, balcoes, n_bilhetes = dimensionar_terminal(php, nivel)
            
            print(f"\n{'='*15} ANO DE PROJETO: {ano} {'='*15}")
            print(f"Demanda Anual Projetada: {d:,.0f} pax/ano")
            print(f"PHP: {php:.2f} pax/hora-pico")
            
            print(f"\n-- DIMENSIONAMENTO DO TERMINAL (NÍVEL {nivel}) --")
            for k, v in areas.items(): print(f"  > {k:<25}: {v:>8.2f} m²")
            print(f"\n>> ÁREA TOTAL: {total:>8.2f} m²")

    except ErroInput as e:
        print("\n*** ERRO DE EXECUÇÃO ***")
        print(e)
    except Exception as e:
        print(f"\n*** ERRO INESPERADO ***\n{e}")
