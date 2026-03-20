# DIMENSIONAMENTO PRELIMINAR
# LEITOR DE INPUT + PROJECAO DE POPULACAO + AREA DO TERMINAL + TAMANHO PISTA DE POUSO

from erros import erro, validar_numero, validar_inteiro, validar_positivo, ErroInput
import numpy as np


# BLOCOS DO INPUT

BLOCOS = ["POPULACAO", "ALTITUDE", "TEMPERATURA", "DECLIVIDADE", "ENVERGADURA", "DADOS_MENSAIS"]

def eh_bloco(linha):
    return linha in BLOCOS


# LEITURA DO INPUT

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
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("POPULACAO", "valor não informado.")
            dados["POPULACAO"] = ler_populacao(linhas[i+1])

        elif linha == "ALTITUDE":
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("ALTITUDE", "valor não informado.")
            dados["ALTITUDE"] = validar_numero(linhas[i+1], "ALTITUDE")

        elif linha == "TEMPERATURA":
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("TEMPERATURA", "valor não informado.")
            dados["TEMPERATURA"] = validar_numero(linhas[i+1], "TEMPERATURA")

        elif linha == "DECLIVIDADE":
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("DECLIVIDADE", "valor não informado.")
            dados["DECLIVIDADE"] = validar_numero(linhas[i+1], "DECLIVIDADE")

        elif linha == "ENVERGADURA":
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("ENVERGADURA", "valor não informado.")

            dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")

        elif linha == "DADOS_MENSAIS":
            if i+1 >= len(linhas) or eh_bloco(linhas[i+1]):
                erro("DADOS_MENSAIS", "valor não informado.")
            dados["DADOS_MENSAIS"] = ler_dados_mensais(linhas, i+1)

        i += 1

    return dados


# POPULACAO

def ler_populacao(linha):

    v = linha.split()

    opcao = validar_inteiro(v[0], "OPCAO")
    intervalo = validar_inteiro(v[1], "INTERVALO")
    n = validar_inteiro(v[2], "NUM_INTERVALOS")

    if len(v) < 3:
        erro("POPULACAO", "dados incompletos.")

    if intervalo * n > 50:
        erro("POPULACAO", "horizonte maior que 50 anos.")

    return [intervalo * i for i in range(1, n+1)]


# DADOS MENSAIS

def ler_dados_mensais(linhas, i):

    dados = []
    linha_num = 1

    while i < len(linhas):

        if eh_bloco(linhas[i]):
            break

        valores = linhas[i].split()

        if not valores:
            erro("DADOS_MENSAIS", f"linha {linha_num} vazia.")

        for v in valores:

            if v == "":
                erro("DADOS_MENSAIS", f"valor vazio na linha {linha_num}.")

            # número
            try:
                num = float(v)
            except:
                erro("DADOS_MENSAIS", f"valor inválido ('{v}') na linha {linha_num}. Digite número.")

            # decimal
            if not num.is_integer():
                erro("DADOS_MENSAIS", f"valor decimal ({v}) na linha {linha_num}. Use inteiros.")

            # negativo
            if num < 0:
                erro("DADOS_MENSAIS", f"valor negativo ({v}) na linha {linha_num}.")

            dados.append(int(num))

        linha_num += 1
        i += 1

    if len(dados) == 0:
        erro("DADOS_MENSAIS", "nenhum dado informado.")

    if len(dados) % 12 != 0:
        erro("DADOS_MENSAIS", f"quantidade inválida ({len(dados)}). Deve ser múltiplo de 12.")

    return dados


# PREVISÃO

def previsao_demanda(dados, periodo=12):

    dados = np.array(dados)

    mm = np.convolve(dados, np.ones(periodo)/periodo, mode='valid')
    dados_corte = dados[periodo-1:]

    isazonal = dados_corte / mm

    sazonal = [np.mean(isazonal[i::periodo]) for i in range(periodo)]
    sazonal = np.array(sazonal)

    sazonal_expandido = np.tile(sazonal, int(np.ceil(len(dados_corte)/len(sazonal))))[:len(dados_corte)]
    tendencia = dados_corte / sazonal_expandido

    t = np.arange(len(tendencia))
    coef = np.polyfit(t, tendencia, 1)

    t_fut = len(tendencia)
    trend_fut = coef[0]*t_fut + coef[1]

    mes = t_fut % periodo
    previsao = trend_fut * sazonal[mes]

    return previsao


# FHP + PHP

def fator_hora_pico(d):

    if d < 100000: return 0.200
    elif d < 500000: return 0.130
    elif d < 1000000: return 0.080
    elif d < 10000000: return 0.068
    elif d < 20000000: return 0.051
    elif d < 30000000: return 0.040
    else: return 0.035


def calcular_php(d):

    fhp = fator_hora_pico(d)
    diaria = d / 365
    php = diaria * fhp

    return php, fhp


# TERMINAL

def dimensionar_terminal(php):

    indices = {
        "Saguão embarque": 1.2,
        "Pré-embarque": 1.0,
        "Check-in": 0.8,
        "Bagagem": 1.5,
        "Desembarque": 1.0
    }

    areas = {k: v*php for k, v in indices.items()}
    total = sum(areas.values())

    return areas, total


# PISTA

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
        if e < 24: return 18
        elif e < 36: return 23

    elif L < 1200:
        if e < 24: return 23
        elif e < 36: return 30

    elif L < 1800:
        if e < 36: return 30
        elif e < 52: return 45

    else:
        if 24 <= e < 65: return 45
        elif 65 <= e <= 80: return 60

    return None

# MAIN

if __name__ == "__main__":

    try:

        dados = ler_arquivo_input("input.txt")

        # PREVISÃO
        prev_mensal = previsao_demanda(dados["DADOS_MENSAIS"])
        demanda_anual = prev_mensal * 12

        # TERMINAL
        php, fhp = calcular_php(demanda_anual)
        areas, total = dimensionar_terminal(php)

        # PISTA
        L0 = obter_L0(dados["ENVERGADURA"])
        Lf = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["DECLIVIDADE"])
        largura = largura_pista(Lf, dados["ENVERGADURA"])

        # SAÍDA
        print("\n==== POPULAÇÃO ====")
        print(dados["POPULACAO"])

        print("\n==== PREVISÃO ====")
        print(f"Mensal: {prev_mensal:.2f}")
        print(f"Anual: {demanda_anual:.2f}")

        print("\n==== TERMINAL ====")
        print(f"PHP: {php:.2f}")
        for k, v in areas.items():
            print(f"{k}: {v:.2f} m²")
        print(f"Total: {total:.2f} m²")

        print("\n==== PISTA ====")
        print(f"Comprimento: {Lf:.2f} m")
        print(f"Largura: {largura if largura else 'Não aplicável'} m")

    except ErroInput as e:
        print("\n*** ERRO NO INPUT ***")
        print(e)
