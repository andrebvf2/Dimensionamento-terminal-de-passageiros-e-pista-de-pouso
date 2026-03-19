# BIBLIOTECA DE ERROS

class ErroInput(Exception):
    pass


def erro(msg):
    raise ErroInput(msg)


def validar_numero(valor, nome_variavel):
    try:
        return float(valor)
    except:
        erro(f"ERRO: Valor inválido para {nome_variavel}. Por favor, inserir valor numérico.")


def validar_inteiro(valor, nome_variavel):
    try:
        if float(valor) % 1 != 0:
            erro(f"ERRO: {nome_variavel} não pode ser número quebrado.")
        return int(valor)
    except:
        erro(f"ERRO: Valor inválido para {nome_variavel}. Deve ser inteiro.")


def validar_nao_nulo(valor, nome_variavel):
    if valor == "" or valor is None:
        erro(f"ERRO: A variável {nome_variavel} não pode ser vazia (null).")


def validar_populacao(valor):
    valor = validar_numero(valor, "POPULACAO")

    if valor < 0:
        erro("ERRO: População não pode ser negativa.")

    if valor % 1 != 0:
        erro("ERRO: População não pode ser número quebrado.")

    return int(valor)
