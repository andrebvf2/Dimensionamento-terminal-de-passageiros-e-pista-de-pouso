# BIBLIOTECA DE ERROS

class ErroInput(Exception):
    pass


def erro(campo, msg):
    raise ErroInput(f"ERRO em {campo}: {msg}")


def validar_numero(valor, campo):

    if str(valor).strip() == "":
        erro(campo, "valor não informado.")

    try:
        return float(valor)
    except ValueError:
        erro(campo, f"valor inválido ('{valor}'). Digite um número.")


def validar_inteiro(valor, campo):

    num = validar_numero(valor, campo)

    if not float(num).is_integer():
        erro(campo, f"valor decimal ({valor}). Use apenas números inteiros.")

    return int(num)


def validar_positivo(valor, campo):

    num = validar_numero(valor, campo)

    if num < 0:
        erro(campo, f"valor negativo ({valor}). Não permitido.")

    return num


def validar_inteiro_positivo(valor, campo):

    num = validar_inteiro(valor, campo)

    if num < 0:
        erro(campo, f"valor negativo ({valor}). Não permitido.")

    return num
