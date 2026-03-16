import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from pathlib import Path

# --- ⚙️ CONFIGURAÇÕES DO EXPERIMENTO ---
NOME_DO_MODELO = './best_1024.pt'                 # Caminho do seu modelo treinado
PASTA_IMAGENS = '../dataset_final/val/images' # Pasta com as imagens de validação (inéditas)
ARQUIVO_GABARITO = '../area.xlsx - Página1.csv' # Seu CSV com as medidas reais

# Configuração da Referência Física (O Quadrado de Papel)
LADO_DA_ESCALA_CM = 4.0                    # Lado do quadrado em cm
AREA_REAL_REFERENCIA_MM2 = (LADO_DA_ESCALA_CM * 10) ** 2  # Converte cm para mm² (1600 mm²)

# Travas de Segurança (Filtros)
LIMITE_MINIMO_PIXELS_ESCALA = 1000         # Ignora escalas menores que isso (ruído/sujeira)
USAR_TTA = False                            # Test Time Augmentation (Melhora precisão, levemente mais lento)
## NOTA: O USSO DE TTA NÃO FOI POSSIVEL POR NAO SER COMPATIVEL COM SEGMENTACAO 

# IDs das Classes no YOLO (Verifique seu data.yaml)
ID_CLASSE_ESCALA = 0
ID_CLASSE_FUNGO = 1
# ----------------------------------------

def carregar_gabarito(caminho_csv):
    """Lê o CSV e cria um dicionário {nome_arquivo: area_real}."""
    print(f"📂 Lendo gabarito: {caminho_csv}...")
    try:
        # Tenta ler detectando automaticamente o separador (vírgula ou ponto e vírgula)
        df = pd.read_csv(caminho_csv, sep=None, engine='python')
        
        # Ajuste aqui os nomes exatos das colunas do seu Excel
        col_id = 'identificador'   
        col_area = 'Area(mm²)'     
        
        gabarito = {}
        for _, row in df.iterrows():
            # Limpa o nome (remove extensão e espaços)
            nome = str(row[col_id]).strip().rsplit('.', 1)[0]
            # Converte '58,97' para float 58.97
            area_str = str(row[col_area]).replace(',', '.')
            gabarito[nome] = float(area_str)
            
        print(f"✅ Gabarito carregado com {len(gabarito)} amostras.")
        return gabarito
    except Exception as e:
        print(f"❌ Erro crítico ao ler CSV: {e}")
        return None

def processar_mascaras(resultado_yolo):
    """Extrai as áreas em pixels da Escala e do Fungo."""
    area_px_escala = 0.0
    area_px_fungo = 0.0

    if resultado_yolo.masks is None:
        return 0, 0

    # Itera sobre cada objeto detectado na imagem
    for i, class_id in enumerate(resultado_yolo.boxes.cls):
        mascara = resultado_yolo.masks.xy[i]
        if len(mascara) == 0: continue
        
        area_poligono = cv2.contourArea(mascara.astype(np.float32))
        classe = int(class_id)

        if classe == ID_CLASSE_ESCALA:
            # Pega a MAIOR escala encontrada (assume que a maior é a correta, ignora ruído)
            if area_poligono > area_px_escala:
                area_px_escala = area_poligono
        
        elif classe == ID_CLASSE_FUNGO:
            # Soma todas as colônias de fungo (caso a IA detecte pedaços separados)
            area_px_fungo += area_poligono

    return area_px_escala, area_px_fungo

def gerar_graficos(df_resultados):
    """Gera gráficos profissionais para o relatório científico."""
    
    # 1. Gráfico de Dispersão (Real vs IA)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_resultados['Real_mm2'], df_resultados['IA_mm2'], 
                alpha=0.6, color='blue', edgecolors='k', label='Amostras')
    
    # Linha Ideal (x=y)
    limite = max(df_resultados['Real_mm2'].max(), df_resultados['IA_mm2'].max())
    plt.plot([0, limite], [0, limite], 'r--', linewidth=2, label='Ideal (Perfeição)')
    
    plt.title(f"Correlação de Mensuração: Método Manual vs IA\n$R^2$ = {df_resultados.attrs['r2']:.4f}")
    plt.xlabel('Área Real Manual ($mm^2$)')
    plt.ylabel('Área Predita IA ($mm^2$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('grafico_correlacao_final.png', dpi=300)
    plt.close()

    # 2. Gráfico de Resíduos (Erro Relativo por Tamanho) - CRUCIAL
    plt.figure(figsize=(10, 6))
    plt.scatter(df_resultados['Real_mm2'], df_resultados['Erro_Perc'], 
                color='purple', alpha=0.6, edgecolors='k')
    
    plt.axhline(0, color='green', linestyle='-', linewidth=1.5)
    plt.axhline(df_resultados['Erro_Perc'].mean(), color='red', linestyle='--', 
                label=f'Erro Médio: {df_resultados["Erro_Perc"].mean():.1f}%')

    plt.title("Análise de Resíduos: Sensibilidade à Escala")
    plt.xlabel("Tamanho da Colônia ($mm^2$)")
    plt.ylabel("Erro Relativo (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('grafico_residuos_erro.png', dpi=300)
    plt.close()
    
    print("\n📊 Gráficos salvos: 'grafico_correlacao_final.png' e 'grafico_residuos_erro.png'")

def main():
    # 1. Preparação
    gabarito = carregar_gabarito(ARQUIVO_GABARITO)
    if not gabarito: return

    print(f"🚀 Carregando modelo YOLO: {NOME_DO_MODELO}")
    model = YOLO(NOME_DO_MODELO)
    
    imagens = list(Path(PASTA_IMAGENS).glob('*.[jp][pn]g')) # Pega .jpg e .png
    resultados = []

    print(f"\n{'ARQUIVO':<30} | {'REAL':<10} | {'IA':<10} | {'ERRO %':<10}")
    print("-" * 70)

    # 2. Processamento
    for img_path in imagens:
        nome_arquivo = img_path.stem # Nome sem extensão
        
        # Verifica se temos o gabarito dessa imagem
        if nome_arquivo not in gabarito:
            continue
            
        area_real = gabarito[nome_arquivo]

        # Inferência (com TTA se ativado)
        # Forçamos imgsz=1024 para ele não comprimir a imagem
        results = model(str(img_path), 
                verbose=False, 
                augment=False,      # Desligamos TTA para não confundir agora
                imgsz=1024,         # Força usar a resolução HD do treino
                retina_masks=True,  # OBRIGATÓRIO: Gera máscaras na resolução nativa
                conf=0.20,          # Baixamos um pouco a confiança para pegar bordas finas
                iou=0.60)           # Ajuste fino de sobreposição
        
        # Extração de Máscaras
        px_escala, px_fungo = processar_mascaras(results[0])

        # 3. Matemática e Filtros
        if px_escala < LIMITE_MINIMO_PIXELS_ESCALA:
            # Se a escala for muito pequena, é ruído ou detecção ruim. Ignora.
            continue
            
        if px_fungo == 0:
            area_ia = 0
        else:
            # Regra de Três: (Area_Ref_mm2 / px_escala) * px_fungo
            fator_conversao = AREA_REAL_REFERENCIA_MM2 / px_escala
            area_ia = px_fungo * fator_conversao

        # Cálculo de Erro
        erro_abs = abs(area_ia - area_real)
        erro_perc = (erro_abs / area_real) * 100 if area_real > 0 else 0

        # Mostra no console
        print(f"{nome_arquivo:<30} | {area_real:<10.2f} | {area_ia:<10.2f} | {erro_perc:<10.2f}")

        resultados.append({
            'Arquivo': nome_arquivo,
            'Real_mm2': area_real,
            'IA_mm2': area_ia,
            'Erro_Perc': erro_perc
        })

    # 4. Consolidação e Relatório
    if not resultados:
        print("⚠️ Nenhuma imagem processada. Verifique nomes e caminhos.")
        return

    df = pd.DataFrame(resultados)
    
    # Métricas Estatísticas
    mape = df['Erro_Perc'].mean()
    r2 = r2_score(df['Real_mm2'], df['IA_mm2'])
    rmse = np.sqrt(mean_squared_error(df['Real_mm2'], df['IA_mm2']))
    
    # Salva métricas no DataFrame para usar nos gráficos
    df.attrs['r2'] = r2

    print("-" * 70)
    print(f"🏆 RESULTADO FINAL ({len(df)} amostras validadas):")
    print(f"🔹 Erro Médio (MAPE): {mape:.2f}%")
    print(f"🔹 Correlação (R²):   {r2:.4f}")
    print(f"🔹 Erro RMSE:         {rmse:.2f}")

    # Salva Excel final
    df.to_csv('relatorio_validacao_final.csv', index=False)
    print("💾 Relatório salvo: 'relatorio_validacao_final.csv'")

    # Gera Gráficos
    gerar_graficos(df)

if __name__ == "__main__":
    main()