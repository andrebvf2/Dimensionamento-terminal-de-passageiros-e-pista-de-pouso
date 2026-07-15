# DIMENSIONAMENTO PRELIMINAR E PREVISÃO DE DEMANDA
# MAIN - ORQUESTRADOR

import math
import os
import logging
import re
import unicodedata
import numpy as np
import pandas as pd

# IMPORTANDO OS NOSSOS MÓDULOS OTIMIZADOS
from terminal import calcular_php, dimensionar_terminal
from pista import calcular_pista, largura_pista, determinar_configuracao_pista, buscar_ventos_local

from erros import ErroInput, erro, validar_numero, validar_inteiro, validar_positivo, validar_inteiro_positivo, validar_quantidade_parametros, validar_opcao

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =========================
# FUNÇÕES DE FORMATAÇÃO
# =========================
# =========================
# FUNÇÕES DE FORMATAÇÃO E TEXTO
# =========================
def formato_br(valor, casas=2):
    if valor is None: return "N/A"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(',', 'X').replace('.', ',').replace('X', '.')

def obter_direcao_rosa_dos_ventos(pista_str):
    # Rosa dos Ventos com 16 direções precisas
    direcoes = [
        "Norte", "Norte-Nordeste", "Nordeste", "Leste-Nordeste", 
        "Leste", "Leste-Sudeste", "Sudeste", "Sul-Sudeste", 
        "Sul", "Sul-Sudoeste", "Sudoeste", "Oeste-Sudoeste", 
        "Oeste", "Oeste-Noroeste", "Noroeste", "Norte-Noroeste"
    ]
    try:
        c1, c2 = pista_str.split('/')
        ang1, ang2 = int(c1) * 10, int(c2) * 10
        dir1 = direcoes[int((ang1 + 11.25) / 22.5) % 16]
        dir2 = direcoes[int((ang2 + 11.25) / 22.5) % 16]
        return f"Eixo {dir1} / {dir2}"
    except:
        return ""

def normalizar(txt):
    if pd.isna(txt): return ""
    txt = re.sub(r'\(.*?\)', '', str(txt))
    txt = re.sub(r'^\d+\s*', '', txt)
    return unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode().lower().strip()

# =========================
# MANIPULAÇÃO DINÂMICA DE DADOS (IBGE E ANAC)
# =========================
def carregar_bases_locais():
    files = ["ibge_limpo.csv", "PIB2023.csv", "anac.csv"]
    for f in files:
        if not os.path.exists(f):
            raise ErroInput("ARQUIVO", f"Arquivo '{f}' não encontrado na pasta atual.")
    try:
        # IBGE População
        df_pop = pd.read_csv("ibge_limpo.csv", encoding='utf-8')
        df_pop["cidade_norm"] = df_pop["cidade"].apply(normalizar)

        # PIB
        df_pib_raw = pd.read_csv("PIB2023.csv", sep=None, engine='python', encoding='utf-8-sig', skiprows=3)
        df_pib_raw.columns = df_pib_raw.columns.str.strip().str.lower()
        col_cid_pib = [c for c in df_pib_raw.columns if 'mun' in c or 'cid' in c][0]
        col_val_pib = [c for c in df_pib_raw.columns if '2023' in c or 'valor' in c or 'pib' in c][0]
        df_pib = pd.DataFrame({
            'cidade_norm': df_pib_raw[col_cid_pib].apply(normalizar),
            'pib_valor': pd.to_numeric(df_pib_raw[col_val_pib], errors='coerce')
        }).dropna()

        # Merge IBGE + PIB
        df_ibge = pd.merge(df_pop, df_pib, on="cidade_norm", how="left")
        df_ibge["pib"] = df_ibge["pib_valor"].fillna(0)

        # ANAC
        df_anac = pd.read_csv("anac.csv", sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
        df_anac.columns = df_anac.columns.str.strip().str.lower()
        col_mun_anac = [c for c in df_anac.columns if 'mun' in c or 'cid' in c][0]
        df_anac = df_anac.rename(columns={col_mun_anac: 'municipio'})
        df_anac["mun_norm"] = df_anac["municipio"].apply(normalizar)

        print(">> Bases locais do IBGE e ANAC consolidadas com sucesso!")
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
    
    # Filtros de anomalias (Cidades turísticas extremas ou municípios de Região Metropolitana que atendem Capitais)
    anomalias_conhecidas = [
        "fernando de noronha", "porto seguro", "gramado", "rio de janeiro", "sao paulo", "belem",
        "rio largo", "sao goncalo do amarante", "confins", "sao jose dos pinhais", "guarulhos", 
        "campinas", "bayeux", "varzea grande", "parnamirim", "ilha do governador", "macae"
    ]
    df_filtrado = df_filtrado[~df_filtrado["cidade_norm"].isin(anomalias_conhecidas)]
    df_filtrado = df_filtrado[(df_filtrado["populacao"] <= pop_alvo * 10) & (df_filtrado["populacao"] >= pop_alvo * 0.1)]
    
    # Margem de busca de 50 cidades
    return df_filtrado.sort_values("score")["cidade_norm"].head(50).tolist()

def obter_series_anac(cidades_norm, df_anac):
    historico = []
    for cidade in cidades_norm:
        df_cidade = df_anac[df_anac["mun_norm"] == cidade]
        if not df_cidade.empty:
            df_cidade = df_cidade.sort_values("ano_mes")
            if len(df_cidade) >= 60:
                # Extrai estritamente os últimos 60 meses registrados
                serie = df_cidade["passageiros"].tail(60).values
                historico.append((cidade, serie.tolist()))
    return historico

# =========================
# O MOTOR DE PREVISÃO (MÉTODO CLÁSSICO DE DECOMPOSIÇÃO + ANCORADO)
# =========================
def prever_demanda_unica_cidade(serie, anos_projecao, ano_base):
    n_meses = 60
    
    # 1. MMC e Índices Sazonais
    m12 = [sum(serie[i:i+12]) / 12 for i in range(n_meses - 11)]
    mmc = [(m12[i] + m12[i+1]) / 2 for i in range(len(m12) - 1)]
    
    is_mensal = [None] * n_meses
    for i in range(len(mmc)):
        mes_idx = i + 6
        if mmc[i] != 0:
            is_mensal[mes_idx] = serie[mes_idx] / mmc[i]
            
    is_medio = []
    for mes in range(12):
        valores_mes = [is_mensal[i] for i in range(mes, n_meses, 12) if is_mensal[i] is not None]
        is_medio.append(sum(valores_mes) / len(valores_mes) if valores_mes else 1.0)
        
    tendencia = []
    for i in range(n_meses):
        mes_idx = i % 12
        val_tendencia = serie[i] / is_medio[mes_idx] if is_medio[mes_idx] != 0 else serie[i]
        tendencia.append(val_tendencia)
        
   # 2. Regressão OLS (Mínimos Quadrados / Estilo Excel - MMQ Livre)
    n = n_meses
    sum_x = sum(range(1, n + 1))
    sum_y = sum(tendencia)
    sum_xy = sum(x * y for x, y in zip(range(1, n + 1), tendencia))
    sum_x2 = sum(x**2 for x in range(1, n + 1))
    
    den = (n * sum_x2 - sum_x**2)
    a = (n * sum_xy - sum_x * sum_y) / den if den != 0 else 0
    b = (sum_y - a * sum_x) / n
    
    equacao = f"Y = {a:.4f} * X {b:+.4f}"
    # 3. Projeção Dinâmica
    resultados = {}
    for ano in anos_projecao:
        mes_inicio = (ano - 2009) * 12 + 1
        mes_fim = mes_inicio + 11
        
        demanda_anual_projetada = 0
        for mes_t in range(mes_inicio, mes_fim + 1):
            nova_tendencia = (a * mes_t) + b
            mes_idx = (mes_t - 1) % 12
            demanda_mes = max(0, nova_tendencia * is_medio[mes_idx])
            demanda_anual_projetada += demanda_mes
            
        resultados[ano] = demanda_anual_projetada
        
    return resultados, equacao

def calcular_demanda_dinamica(cidade_alvo, anos, ano_base):
    df_ibge, df_anac = carregar_bases_locais()
    
    pop_alvo, pib_alvo = get_dados_ibge(cidade_alvo, df_ibge)
    print(f"\n>> Perfil Alvo: {cidade_alvo.title()} | Pop: {formato_br(pop_alvo, 0)} | PIB: R$ {formato_br(pib_alvo, 2)}")
    
    cidades_similares = selecionar_cidades_similares(pop_alvo, pib_alvo, df_ibge, df_anac)
    series_com_nomes = obter_series_anac(cidades_similares, df_anac)
    
    if len(series_com_nomes) == 0: 
        raise ErroInput("DADOS", "Nenhuma cidade similar possui o histórico de 60 meses.")
        
    print(f"\n--- EQUAÇÕES DE TENDÊNCIA (FILTRO DE CRESCIMENTO ATIVADO) ---")
    
    resultado_final = {ano: 0 for ano in anos}
    cidades_aprovadas = 0
    
    # MUDANÇA: Loop inteligente que testa se a reta está subindo ou caindo
    for cid_similar, serie in series_com_nomes:
        if cidades_aprovadas >= 5:
            break # Já achou 5 cidades perfeitamente saudáveis, pode parar
            
        projecao_similar, equacao_excel = prever_demanda_unica_cidade(serie, anos, ano_base)
        
        # Só aprova se a demanda do último ano projetado for maior que a do primeiro
        if projecao_similar[anos[-1]] > projecao_similar[anos[0]]:
            print(f"[{cid_similar.title():<22}] {equacao_excel} (APROVADA)")
            
            pop_similar, _ = get_dados_ibge(cid_similar, df_ibge)
            
            for ano in anos:
                taxa_per_capita = projecao_similar[ano] / pop_similar
                demanda_alvo_projetada = taxa_per_capita * pop_alvo
                resultado_final[ano] += demanda_alvo_projetada
                
            cidades_aprovadas += 1
        else:
            print(f"[{cid_similar.title():<22}] {equacao_excel} (DESCARTADA: Em Queda)")
            
    if cidades_aprovadas == 0:
        raise ErroInput("DADOS", "As cidades similares só apresentaram retração. Tente outro modelo.")
        
    # Tira a média usando apenas a quantidade exata de cidades aprovadas
    for ano in anos:
        resultado_final[ano] /= cidades_aprovadas
        
    return resultado_final

# =========================
# LEITURA DE INPUT
# =========================
def ler_populacao(linha):
    # Usando o validador de quantidade!
    v = validar_quantidade_parametros(linha.split(), 3, "POPULACAO")
    
    ano_inicio = validar_inteiro(v[0], "ANO_INICIO")
    intervalo = validar_inteiro(v[1], "INTERVALO")
    n = validar_inteiro_positivo(v[2], "NUM_INTERVALOS")
    
    # O 'range(n)' começa em 0. 
    # Assim, o primeiro ano projetado será o próprio ano_inicio (ex: 2024 + 5*0 = 2024)
    return [ano_inicio + (intervalo * i) for i in range(n)], ano_inicio
def ler_arquivo_input(caminho):
    dados = {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = [l.strip() for l in f if l.strip() != ""]
    except FileNotFoundError: 
        erro("ARQUIVO", "input.txt não encontrado.")
        
    i = 0
    while i < len(linhas):
        linha = linhas[i].upper()
        if linha == "POPULACAO": 
            dados["ANOS_PROJ"], dados["ANO_ZERO"] = ler_populacao(linhas[i+1])
        elif linha == "ALTITUDE": 
            dados["ALTITUDE"] = validar_numero(linhas[i+1], "ALTITUDE")
        elif linha == "TEMPERATURA": 
            dados["TEMPERATURA"] = validar_numero(linhas[i+1], "TEMPERATURA")
        elif linha == "COTAS_PISTA":
            # AQUI: Usando o validador de quantidade!
            v = validar_quantidade_parametros(linhas[i+1].split(), 2, "COTAS_PISTA")
            dados["COTA_ALTA"] = validar_numero(v[0], "COTA_ALTA")
            dados["COTA_BAIXA"] = validar_numero(v[1], "COTA_BAIXA")
        elif linha == "ENVERGADURA": 
            dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")
        elif linha == "COMPRIMENTO_BASICO": 
            dados["COMPRIMENTO_BASICO"] = validar_positivo(linhas[i+1], "COMPRIMENTO_BASICO")
        elif linha == "DEMANDA_ANUAL":
            valor = linhas[i+1].upper()
            if "CALCULAR" in valor:
                partes = validar_quantidade_parametros(valor.split(), 2, "DEMANDA_ANUAL")
                dados["CIDADE_ALVO"] = " ".join(partes[1:]).title() 
                dados["DEMANDA_ANUAL"] = "CALCULAR"
            else: 
                dados["DEMANDA_ANUAL"] = validar_positivo(valor, "DEMANDA_ANUAL")
        elif linha == "NIVEL_SERVICO":
            # Usando o validador de opções!
            dados["NIVEL_SERVICO"] = validar_opcao(linhas[i+1], ["A", "B", "C"], "NIVEL_SERVICO")
        i += 1
    return dados

# =========================
# MAIN - ORQUESTRADOR
# =========================
if __name__ == "__main__":
    try:
        dados = ler_arquivo_input("input.txt")
        anos = dados.get("ANOS_PROJ", [2014, 2019, 2029, 2039])
        ano_zero = dados.get("ANO_ZERO", 2013)
        nivel = dados.get("NIVEL_SERVICO", "B")
        
        if "COMPRIMENTO_BASICO" not in dados: raise ErroInput("COMPRIMENTO_BASICO", "Falta o bloco COMPRIMENTO_BASICO no input.txt")
        if "COTA_ALTA" not in dados or "COTA_BAIXA" not in dados: raise ErroInput("COTAS_PISTA", "O bloco COTAS_PISTA não foi encontrado.")
        
        L0 = dados["COMPRIMENTO_BASICO"]
        
        # --- MÓDULO PISTA ---
        Lf, ca, ct, cd, decl_calculada = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["COTA_ALTA"], dados["COTA_BAIXA"])
        largura = largura_pista(L0, dados["ENVERGADURA"])

        print("\n==== INFRAESTRUTURA DA PISTA ====")
        print(f"Comprimento Básico (L0): {formato_br(L0)} m")
        print(f"Cotas Topográficas (Alta/Baixa): {formato_br(dados['COTA_ALTA'])}m | {formato_br(dados['COTA_BAIXA'])}m")
        print(f"Declividade Calculada: {formato_br(decl_calculada)}%")
        print(f"Fatores - CA: {formato_br(ca, 4)} | CT: {formato_br(ct, 4)} | CD: {formato_br(cd, 4)}")
        print(f"Comprimento Corrigido (Lf): {formato_br(Lf)} m")
        print(f"Largura da Pista: {formato_br(largura, 0) if largura else 'Fora das especificações'} m")

       
       # --- ANÁLISE DE VENTOS (TABELA 5 do TCC) ---
        if L0 < 1200: limite_vento = 5.15
        elif 1200 <= L0 < 1500: limite_vento = 6.70
        else: limite_vento = 10.30
            
        df_ventos_local = buscar_ventos_local("historico_ventos.csv")
        pista_ideal, cobertura, secundaria = determinar_configuracao_pista(df_ventos_local, limite_vento)
        
        #Chamando a função para traduzir a Rosa dos Ventos
        sentido_rosa_ventos = obter_direcao_rosa_dos_ventos(pista_ideal)
        
        print("\n==== LOCAÇÃO E GEOMETRIA DA PISTA ====")
        print(f"Orientação Magnética: Cabeceiras {pista_ideal} ({sentido_rosa_ventos})")
        print(f"Cobertura de Vento Calculada: {formato_br(cobertura)}% do tempo operável")

        if secundaria: print(">>> RECOMENDAÇÃO: O projeto exigirá pistas transversais (cobertura < 95%).")
        else: print(">>> CONFIGURAÇÃO: Pista única atende aos requisitos (> 95%).")

        # --- DEMANDA E TERMINAL ---
        if dados.get("DEMANDA_ANUAL") == "CALCULAR":
            print(f"\n>> Iniciando processamento de séries temporais...")
            demandas_anuais = calcular_demanda_dinamica(dados["CIDADE_ALVO"], anos, ano_zero)
        else:
            demandas_anuais = {ano: dados["DEMANDA_ANUAL"] for ano in anos}

        for ano in anos:
            d = demandas_anuais[ano]
            
            # --- MÓDULO TERMINAL ---
            php, fhp = calcular_php(d)
            areas, total, balcoes, n_bilhetes = dimensionar_terminal(php, nivel)
            
            print(f"\n{'='*15} ANO DE PROJETO: {ano} {'='*15}")
            print(f"Demanda Anual Média (Per Capita Projetada): {formato_br(d, 0)} passageiros")
            print(f"PHP: {formato_br(php)} pax/hora-pico")
            
            print(f"\n-- DIMENSIONAMENTO DO TERMINAL (NÍVEL {nivel}) --")
            for k, v in areas.items(): print(f"  > {k:<25}: {formato_br(v)} m²")
            print(f"\n>> ÁREA TOTAL: {formato_br(total)} m²")

    except ErroInput as e: 
        print(f"\n*** ERRO DE EXECUÇÃO ***\n{e}")
    except Exception as e: 
        print(f"\n*** ERRO INESPERADO ***\n{e}")
