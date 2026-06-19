import math

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
