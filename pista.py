import math
import pandas as pd
import numpy as np

# =======================================================
# MÓDULO 1: GEOMETRIA E DIMENSIONAMENTO DA PISTA
# =======================================================
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


# =======================================================
# MÓDULO 2: ANÁLISE DE VENTOS (ARQUIVO HORÁRIO BRUTO)
# =======================================================
def limpar_valor(v):
    if pd.isna(v) or str(v).strip() == '-' or str(v).strip() == '':
        return 0.0
    try:
        return float(str(v).replace(',', '.'))
    except ValueError:
        return np.nan

def buscar_ventos_local(caminho_arquivo):
    print("\n>> Processando milhares de registros horários de vento...")
    
    # 1. RADAR DE CABEÇALHO: Descobre dinamicamente em qual linha estão as colunas
    skip_lines = 0
    try:
        with open(caminho_arquivo, 'r', encoding='latin1') as f:
            for i, linha in enumerate(f):
                linha_upper = linha.upper()
                if 'DIR' in linha_upper or 'VENTO' in linha_upper or 'WD' in linha_upper:
                    skip_lines = i
                    break
    except Exception:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            for i, linha in enumerate(f):
                if 'DIR' in linha.upper() or 'VENTO' in linha.upper() or 'WD' in linha.upper():
                    skip_lines = i
                    break

    # 2. LÊ O ARQUIVO EXATAMENTE NA LINHA CORRETA
    try:
        df_completo = pd.read_csv(caminho_arquivo, sep=';', skiprows=skip_lines, encoding='latin1', engine='python')
    except Exception:
        df_completo = pd.read_csv(caminho_arquivo, sep=';', skiprows=skip_lines, encoding='utf-8', engine='python')

    # 3. ENCONTRANDO AS COLUNAS (Ignorando a coluna de rajada)
    col_dir = None
    col_vel = None

    for col in df_completo.columns:
        col_up = str(col).upper().strip()
        
        # Procurando a Direção
        if ('DIR' in col_up or 'WD' in col_up) and 'RAJADA' not in col_up:
            col_dir = col
            
        # Procurando a Velocidade (ignorando a rajada máxima)
        if ('VEL' in col_up or 'WS' in col_up or ('VENTO' in col_up and 'M/S' in col_up)) and 'RAJADA' not in col_up:
            col_vel = col

    if col_dir is None or col_vel is None:
        print(f"\n[!] Colunas encontradas no arquivo: {list(df_completo.columns)}")
        raise RuntimeError("Não consegui identificar as colunas de Vento e Direção. Verifique a lista impressa acima.")

    # 4. FILTRA APENAS AS COLUNAS QUE IMPORTAM E LIMPA DADOS INVÁLIDOS
    df_ventos = df_completo[[col_dir, col_vel]].copy()
    df_ventos.columns = ['direcao', 'velocidade']
    
    df_ventos['direcao'] = df_ventos['direcao'].apply(limpar_valor)
    df_ventos['velocidade'] = df_ventos['velocidade'].apply(limpar_valor)
    
    # Remove linhas onde o sensor estava quebrado (NaN)
    df_ventos = df_ventos.dropna(subset=['direcao', 'velocidade'])
    df_ventos = df_ventos[(df_ventos['direcao'] >= 0) & (df_ventos['direcao'] <= 360)]
    df_ventos = df_ventos[df_ventos['velocidade'] >= 0]
    
    print(f">> Sucesso! {len(df_ventos)} registros de vento validados e prontos para cálculo.")
    return df_ventos

def gerar_tabela_frequencia(df_ventos):
    print("\n>> Analisando ventos e gerando Tabela de Frequência...")
    
    bins_vel = [0, 4, 7, 9, 12, 16, np.inf]
    labels_vel = ['0-4 m/s', '4-7 m/s', '7-9 m/s', '9-12 m/s', '12-16 m/s', '>16 m/s']
    
    df = df_ventos.copy()
    df['Classe_Vel'] = pd.cut(df['velocidade'], bins=bins_vel, labels=labels_vel, right=False)
    
    def graus_para_rumo(graus):
        rumos = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int(np.floor(((graus + 11.25) % 360) / 22.5))
        return rumos[idx]

    df['Rumo'] = df['direcao'].apply(graus_para_rumo)
    df.loc[df['velocidade'] < 0.5, 'Rumo'] = 'CALMARIA'
    
    ordem_linhas = ['CALMARIA', 'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    df['Rumo'] = pd.Categorical(df['Rumo'], categories=ordem_linhas, ordered=True)
    tabela = pd.crosstab(df['Rumo'], df['Classe_Vel'], dropna=False)
    
    tabela_pct = (tabela / len(df)) * 100
    tabela_pct['TOTAL (%)'] = tabela_pct.sum(axis=1)
    
    direcoes_cardeais = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                         'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    totais_direcoes = tabela_pct.loc[direcoes_cardeais, 'TOTAL (%)']
    direcao_predominante = totais_direcoes.idxmax()
    porcentagem_predominante = totais_direcoes.max()
    
    linha_predominante = tabela_pct.loc[direcao_predominante].drop('TOTAL (%)')
    velocidade_predominante = linha_predominante.idxmax()
    pct_velocidade = linha_predominante.max()
    
    print(f"\n=======================================================")
    print(f"⭐ ANÁLISE DO VENTO PREDOMINANTE:")
    print(f"   • Direção: {direcao_predominante} ({porcentagem_predominante:.2f}% do tempo)")
    print(f"   • Velocidade mais comum: {velocidade_predominante}")
    print(f"   • Somente a faixa de {velocidade_predominante} em {direcao_predominante} representa {pct_velocidade:.2f}% de todos os ventos!")
    print(f"=======================================================\n")

    tabela_pct.loc['TOTAL (%)'] = tabela_pct.sum(axis=0)
    
    if tabela_pct['>16 m/s']['TOTAL (%)'] == 0:
        tabela_pct = tabela_pct.drop(columns=['>16 m/s'], errors='ignore')
        
    def formatar_celula(valor):
        if pd.isna(valor) or valor == 0:
            return '-'
        else:
            return f"{valor:.4f}".replace('.', ',')
            
    tabela_formatada = tabela_pct.apply(lambda col: col.map(formatar_celula))
    
    nome_saida = "Tabela_12_Frequencia_Ventos.csv"
    tabela_formatada.to_csv(nome_saida, sep=';', encoding='utf-8-sig')
    
    print(f">> Sucesso! Tabela formatada com todas as velocidades salva em: {nome_saida}")
    return tabela_formatada

def determinar_configuracao_pista(df_ventos, limite_vc=6.7):
    print(">> Calculando orientação ideal da pista usando todos os registros brutos...")
    
    melhor_pista = None
    melhor_cobertura = -1.0
    
    velocidades = df_ventos['velocidade'].values
    angulos = df_ventos['direcao'].values
    total_registros = len(df_ventos)

    for pista_ang in range(10, 181, 10):
        diff = np.abs(angulos - pista_ang) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        diff = np.where(diff > 90, 180 - diff, diff)
        
        vc = velocidades * np.sin(np.radians(diff))
        
        ventos_cobertos = np.sum(vc <= limite_vc)
        cobertura = (ventos_cobertos / total_registros) * 100
        
        if cobertura > melhor_cobertura:
            melhor_cobertura = cobertura
            pista_1 = int(pista_ang / 10)
            pista_2 = pista_1 + 18
            melhor_pista = f"{pista_1:02d}/{pista_2:02d}"

    precisa_secundaria = melhor_cobertura < 95.0
    return melhor_pista, melhor_cobertura, precisa_secundaria
