# DIMENSIONAMENTO PRELIMINAR
# LEITOR DE INPUT + PROJECAO DE POPULACAO + AREA DO TERMINAL + TAMANHO PISTA DE POUSO

from erros import erro, validar_numero, validar_inteiro, validar_positivo, ErroInput
import numpy as np
import math

# BLOCOS DO INPUT

BLOCOS = ["POPULACAO", "ALTITUDE", "TEMPERATURA", "DECLIVIDADE", "ENVERGADURA", "DEMANDA_ANUAL", "NIVEL_SERVICO", "TIPO_AEROPORTO"]

def eh_bloco(linha):
    return linha in BLOCOS

# POPULACAO

def ler_populacao(linha):
    v = linha.split()

    if len(v) < 3:
        erro("POPULACAO", "dados incompletos.")

    opcao = validar_inteiro(v[0], "OPCAO")
    intervalo = validar_inteiro(v[1], "INTERVALO")
    n = validar_inteiro(v[2], "NUM_INTERVALOS")

    if intervalo * n > 50:
        erro("POPULACAO", "horizonte maior que 50 anos.")

    return [intervalo * i for i in range(1, n+1)]

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
            dados["POPULACAO"] = ler_populacao(linhas[i+1])

        elif linha == "ALTITUDE":
            dados["ALTITUDE"] = validar_numero(linhas[i+1], "ALTITUDE")

        elif linha == "TEMPERATURA":
            dados["TEMPERATURA"] = validar_numero(linhas[i+1], "TEMPERATURA")

        elif linha == "DECLIVIDADE":
            dados["DECLIVIDADE"] = validar_numero(linhas[i+1], "DECLIVIDADE")

        elif linha == "ENVERGADURA":
            dados["ENVERGADURA"] = validar_positivo(linhas[i+1], "ENVERGADURA")

        elif linha == "DEMANDA_ANUAL":
            dados["DEMANDA_ANUAL"] = validar_positivo(linhas[i+1], "DEMANDA_ANUAL")

        elif linha == "NIVEL_SERVICO":
            nivel = linhas[i+1].upper()
            if nivel not in ["A", "B", "C"]:
                erro("NIVEL_SERVICO", "use A, B ou C.")
            dados["NIVEL_SERVICO"] = nivel

        elif linha == "TIPO_AEROPORTO":
            tipo = linhas[i+1].upper()
            if tipo not in ["INT", "DOM", "REG"]:
                erro("TIPO_AEROPORTO", "use INT, DOM ou REG.")
            dados["TIPO_AEROPORTO"] = tipo

        i += 1

    return dados


# PHP (ANAC)

def fator_hora_pico(d):
    if d < 100000: return 0.169
    elif d < 500000: return 0.068
    elif d < 1000000: return 0.064
    elif d < 10000000: return 0.027
    else: return 0.024

def calcular_php(d):
    fhp = fator_hora_pico(d)
    php = d * fhp/100
    return php, fhp


# TERMINAL COMPLETO

def dimensionar_terminal(php, nivel, tipo):

    # SAGUÃO EMBARQUE 
    indices_embarque = {
        "A": {"INT": 2.50, "DOM": 2.20, "REG": 1.80},
        "B": {"INT": 2.00, "DOM": 1.80, "REG": 1.50},
        "C": {"INT": 1.60, "DOM": 1.40, "REG": 1.20}
    }
    area_saguao = php * indices_embarque[nivel][tipo]

    # PRÉ-EMBARQUE 
    indices_pre = {
        "A": {"INT": 1.60, "DOM": 1.40, "REG": 1.20},
        "B": {"INT": 1.40, "DOM": 1.20, "REG": 1.00},
        "C": {"INT": 1.10, "DOM": 1.00, "REG": 0.80}
    }
    area_pre = php * indices_pre[nivel][tipo]

    # CHECK-IN
    pax_checkin = 0.20 * php
    balcoes = math.ceil(pax_checkin / 7.5)

    area_balcao = {"A": 24.12, "B": 19.20, "C": 15.21}
    area_checkin = balcoes * area_balcao[nivel]

    # BILHETES
    percentual_bilhetes = {"A": 0.35, "B": 0.25, "C": 0.15}
    area_bilhete_unit = {"A": 6.48, "B": 5.70, "C": 5.04}

    n_bilhetes = math.ceil(balcoes * percentual_bilhetes[nivel])
    if n_bilhetes < 1:
        n_bilhetes = 1

    area_bilhetes = n_bilhetes * area_bilhete_unit[nivel]

    # SEGURANÇA
    modulos = math.ceil(php / 180)
    area_seguranca = modulos * 13.5

    # TRIAGEM 

    indice_triagem = {
        "INT": 40,
        "DOM": 40,
        "REG": 20
    }

    # número de voos (70 pax por voo)
    n_voos = math.ceil(php / 70)

    # área total da triagem
    area_triagem = n_voos * indice_triagem[tipo]

    # RESTITUIÇÃO 
    indices_bagagem = {
        "A": {"INT": 2.00, "DOM": 1.60, "REG": 1.30},
        "B": {"INT": 1.60, "DOM": 1.40, "REG": 1.10},
        "C": {"INT": 1.30, "DOM": 1.10, "REG": 0.80}
    }
    area_restituicao = php * indices_bagagem[nivel][tipo]

    # DESEMBARQUE 
    indices_desembarque = {
        "A": {"INT": 2.00, "DOM": 1.80, "REG": 1.60},
        "B": {"INT": 1.80, "DOM": 1.50, "REG": 1.20},
        "C": {"INT": 1.50, "DOM": 1.20, "REG": 1.00}
    }
    area_desembarque = php * indices_desembarque[nivel][tipo]

    areas = {
        "Saguão embarque": area_saguao,
        "Pré-embarque": area_pre,
        "Check-in": area_checkin,
        "Bilhetes": area_bilhetes,
        "Segurança": area_seguranca,
        "Triagem e despacho": area_triagem,
        "Restituição bagagem": area_restituicao,
        "Saguão desembarque": area_desembarque
    }

    total = sum(areas.values())

    return areas, total, balcoes, n_bilhetes


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

        demanda = dados["DEMANDA_ANUAL"]
        nivel = dados["NIVEL_SERVICO"]
        tipo = dados["TIPO_AEROPORTO"]

        php, fhp = calcular_php(demanda)

        areas, total, balcoes, n_bilhetes = dimensionar_terminal(php, nivel, tipo)

        L0 = obter_L0(dados["ENVERGADURA"])
        Lf = calcular_pista(L0, dados["ALTITUDE"], dados["TEMPERATURA"], dados["DECLIVIDADE"])
        largura = largura_pista(Lf, dados["ENVERGADURA"])

        # SAÍDA
        print("\n==== POPULAÇÃO ====")
        print(dados["POPULACAO"])

        print("\n==== PISTA ====")
        print(f"Comprimento: {Lf:.2f} m")
        print(f"Largura: {largura if largura else 'Não aplicável'} m")

        print("\n==== DEMANDA ====")
        print(f"Anual: {demanda}")
        print(f"FHP: {fhp}")
        print(f"PHP: {php:.2f}")

        print("\n==== TERMINAL ====")
        print(f"Balcões check-in: {balcoes}")
        print(f"Balcões bilhetes: {n_bilhetes}")

        for k, v in areas.items():
            print(f"{k}: {v:.2f} m²")

        print(f"\nTotal: {total:.2f} m²")

    except ErroInput as e:
        print("\n*** ERRO NO INPUT ***")
        print(e)
