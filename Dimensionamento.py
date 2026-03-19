# DIMENSIONAMENTO PRELIMINAR
# LEITOR DE INPUT + PROJECAO DE POPULACAO

# main.py

from erros import erro, validar_numero, validar_inteiro, ErroInput


# =========================
# LEITURA DO ARQUIVO
# =========================

def ler_arquivo_input(caminho_arquivo):

    dados = {}

    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
            linhas = [linha.strip() for linha in arquivo if linha.strip() != ""]
    except FileNotFoundError:
        erro("ERRO: Arquivo input.txt não encontrado.")

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

        i += 1

    return dados


# =========================
# POPULACAO
# =========================

def ler_populacao(linha):

    valores = linha.split()

    if len(valores) != 3:
        erro("ERRO: A linha de população deve conter 3 valores: OPCAO INTERVALO NUM_INTERVALOS")

    opcao = validar_inteiro(valores[0], "OPCAO")
    intervalo = validar_inteiro(valores[1], "INTERVALO")
    num_intervalos = validar_inteiro(valores[2], "NUM_INTERVALOS")

    if opcao not in [1, 2]:
        erro("ERRO: OPCAO inválida. Use 1 ou 2.")

    if intervalo * num_intervalos > 50:
        erro("ERRO: O horizonte de projeto não pode ultrapassar 50 anos.")

    return gerar_vetor_anos(opcao, intervalo, num_intervalos)


# =========================
# GERAR ANOS
# =========================

def gerar_vetor_anos(opcao, intervalo, num_intervalos):

    if opcao == 2:
        return [intervalo * i for i in range(1, num_intervalos + 1)]

    elif opcao == 1:
        return []


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    try:

        dados = ler_arquivo_input("input.txt")

        print("\n==== DADOS LIDOS DO INPUT ====")
        print("Anos:", dados["POPULACAO"])
        print("Altitude:", dados["ALTITUDE"])
        print("Temperatura:", dados["TEMPERATURA"])
        print("Declividade:", dados["DECLIVIDADE"])

        print("\n==== LOOP DE CALCULO ====")

        for ano in dados["POPULACAO"]:
            print(f"Calculando para {ano} anos...")

    except ErroInput as e:
        print("\n*** ERRO NO INPUT ***")
        print(e)
