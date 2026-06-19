import math
import pandas as pd
import numpy as np
import re
import unicodedata

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

def buscar_ventos_local(nome_arquivo="historico_ventos.csv"):
    print(f"\n>> Lendo base de dados local: {nome_arquivo}...")
    try:
        with open(nome_arquivo, 'r', encoding='latin-1') as f:
            linhas = f.readlines()
            
        linha_cabecalho = 0
        for i, linha in enumerate(linhas[:20]):
            if 'data' in linha.lower() or 'vento' in linha.lower():
                linha_cabecalho = i
                break

        df_completo = pd.read_csv(nome_arquivo, sep=';', skiprows=linha_cabecalho, encoding='latin-1')
        
        def limpar_coluna(nome):
            texto = unicodedata.normalize('NFKD', str(nome)).encode('ascii', 'ignore').decode().lower()
            return re.sub(r'[^a-z]', '', texto)
            
        df_completo.columns = [limpar_coluna(c) for c in df_completo.columns]
        
        col_dir = next((c for c in df_completo.columns if 'dir' in c and 'vento' in c), None)
        col_vel = next((c for c in df_completo.columns if 'vel' in c and 'vento' in c), None)
        
        if not col_dir or not col_vel:
            raise ValueError("Colunas de vento não encontradas mesmo após a limpeza.")
        
        df_ventos = pd.DataFrame()
        df_ventos['direcao'] = df_completo[col_dir].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
        df_ventos['velocidade'] = df_completo[col_vel].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
        
        df_ventos = df_ventos.dropna()
        
        print(f">> Sucesso! {len(df_ventos)} registros reais de vento prontos para cálculo.")
        return df_ventos

    except Exception as e:
        print(f"\n[!] Erro na leitura do arquivo: {e}")
        print(">> Gerando ventos simulados de segurança para o programa não travar...")
        return pd.DataFrame({'direcao': np.random.randint(0, 360, 1000), 'velocidade': np.random.uniform(2, 25, 1000)})
