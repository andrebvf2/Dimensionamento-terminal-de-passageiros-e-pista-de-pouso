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

def validar_quantidade_parametros(lista_valores, quantidade_esperada, campo):
    """Garante que o usuário digitou a quantidade certa de valores na linha (ex: cotas e população)."""
    if len(lista_valores) < quantidade_esperada:
        erro(campo, f"dados incompletos. Esperado {quantidade_esperada} valores, mas recebeu {len(lista_valores)}.")
    return lista_valores

def validar_opcao(valor, opcoes_validas, campo):
    """Garante que o texto digitado está dentro de uma lista de opções permitidas."""
    valor_limpo = str(valor).strip().upper()
    if valor_limpo not in opcoes_validas:
        erro(campo, f"opção '{valor}' inválida. Escolha entre: {opcoes_validas}")
    return valor_limpo
