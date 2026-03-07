import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# --- ⚙️ CONFIGURAÇÕES ---
PASTA_IMAGENS = './dataset_bruto/images'  # A sua pasta com as imagens originais (.jpg ou .png)
PASTA_LABELS = './dataset_bruto/labels'   # A sua pasta com os .txt exportados pelo Label Studio
NOME_ARQUIVO_SAIDA = 'area_gabarito_gerado.csv'

# Configuração da Referência Física
LADO_DA_ESCALA_CM = 4.0
AREA_REAL_REFERENCIA_MM2 = (LADO_DA_ESCALA_CM * 10) ** 2  # 1600 mm²

# IDs das Classes (Conforme anotou no Label Studio)
ID_CLASSE_ESCALA = 0
ID_CLASSE_FUNGO = 1
# ------------------------

def ler_coordenadas_yolo(linha_txt, largura_img, altura_img):
    """
    Transforma uma linha do .txt do YOLO (normalizada 0-1) 
    num polígono de píxeis reais para o OpenCV calcular a área.
    """
    valores = linha_txt.strip().split()
    classe = int(valores[0])
    
    # Pega em todos os números depois da classe e agrupa de 2 em 2 (X e Y)
    coordenadas_normalizadas = np.array(valores[1:], dtype=np.float32).reshape(-1, 2)
    
    # Desnormaliza: multiplica o X pela largura e o Y pela altura
    coordenadas_reais = coordenadas_normalizadas * np.array([largura_img, altura_img])
    
    # Converte para inteiro (o OpenCV exige números inteiros para desenhar o contorno)
    poligono = coordenadas_reais.astype(np.int32)
    
    return classe, poligono

def main():
    print("🚀 A iniciar a geração automática do Gabarito (Ground Truth)...")
    
    caminhos_labels = list(Path(PASTA_LABELS).glob('*.txt'))
    resultados = []
    
    if not caminhos_labels:
        print(f"⚠️ Nenhum ficheiro .txt encontrado na pasta: {PASTA_LABELS}")
        return

    print(f"\n{'FICHEIRO':<30} | {'ÁREA FUNGO (mm²)':<15}")
    print("-" * 50)

    for txt_path in caminhos_labels:
        nome_base = txt_path.stem  # Nome sem o .txt
        
        # Procura a imagem correspondente (pode ser jpg ou png)
        img_path_jpg = Path(PASTA_IMAGENS) / f"{nome_base}.jpg"
        img_path_png = Path(PASTA_IMAGENS) / f"{nome_base}.png"
        
        caminho_img_valido = img_path_jpg if img_path_jpg.exists() else img_path_png
        
        if not caminho_img_valido.exists():
            print(f"⚠️ Imagem não encontrada para {nome_base}. A ignorar...")
            continue
            
        # Lê a imagem APENAS para descobrir a largura e altura
        img = cv2.imread(str(caminho_img_valido))
        altura, largura = img.shape[:2]
        
        area_px_escala = 0.0
        area_px_fungo = 0.0
        
        # Lê as anotações do ficheiro de texto
        with open(txt_path, 'r') as f:
            linhas = f.readlines()
            
        for linha in linhas:
            if not linha.strip(): continue
            
            classe, poligono = ler_coordenadas_yolo(linha, largura, altura)
            area_poligono = cv2.contourArea(poligono)
            
            if classe == ID_CLASSE_ESCALA:
                # Se desenhou mais de uma escala sem querer, guarda a maior
                if area_poligono > area_px_escala:
                    area_px_escala = area_poligono
                    
            elif classe == ID_CLASSE_FUNGO:
                # Soma todas as colónias anotadas
                area_px_fungo += area_poligono
                
        # Validações de segurança
        if area_px_escala == 0:
            print(f"⚠️ Nenhuma escala encontrada em {nome_base}. Área = 0")
            area_final_mm2 = 0.0
        else:
            # A grande conversão (Regra de Três)
            fator_conversao = AREA_REAL_REFERENCIA_MM2 / area_px_escala
            area_final_mm2 = area_px_fungo * fator_conversao
            
        print(f"{nome_base:<30} | {area_final_mm2:<15.2f}")
        
        resultados.append({
            'identificador': nome_base,
            'Area(mm²)': round(area_final_mm2, 2)
        })

    # Exporta para CSV
    df = pd.DataFrame(resultados)
    df.to_csv(NOME_ARQUIVO_SAIDA, index=False)
    
    print("-" * 50)
    print(f"✅ Sucesso! O gabarito com {len(df)} amostras foi salvo como '{NOME_ARQUIVO_SAIDA}'.")
    print("Pode agora usá-lo como o seu 'area.xlsx' oficial!")

if __name__ == "__main__":
    main()