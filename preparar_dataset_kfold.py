import os
import shutil
import random
import yaml
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURAÇÕES DO K-FOLD (LOCAL)
# ==========================================
PASTA_ORIGEM_IMAGENS = "./dataset_bruto/images"
PASTA_ORIGEM_LABELS = "./dataset_bruto/labels"
PASTA_KFOLD_BASE = "./dataset_kfold"          # Pasta que será enviada ao Colab

K_FOLDS = 5                                   
SEED = 42                                     
NOMES_CLASSES = ['escala', 'fungo']
# ==========================================

def limpar_diretorio(caminho):
    if os.path.exists(caminho):
        shutil.rmtree(caminho)

def gerar_yaml_fold_colab(caminho_pasta_fold, fold_idx):
    """
    Gera o data.yaml com caminhos absolutos baseados no padrão do Google Colab.
    Assumimos que você fará o upload e descompactará a pasta direto em /content/
    """
    caminho_yaml = os.path.join(caminho_pasta_fold, f"data_fold_{fold_idx}.yaml")
    
    # Caminho base que a pasta terá quando estiver lá no Colab
    caminho_base_colab = f"/content/dataset_kfold/fold_{fold_idx}"
    
    dados_yaml = {
        'path': caminho_base_colab,
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: nome for i, nome in enumerate(NOMES_CLASSES)}
    }
    
    with open(caminho_yaml, 'w') as f:
        yaml.dump(dados_yaml, f, default_flow_style=False, sort_keys=False)

def main():
    arquivos_imagem = list(Path(PASTA_ORIGEM_IMAGENS).glob('*.[jp][pn]g'))
    if not arquivos_imagem:
        print("❌ Erro: Nenhuma imagem encontrada.")
        return

    nomes_arquivos = [arquivo.stem for arquivo in arquivos_imagem]
    
    random.seed(SEED)
    random.shuffle(nomes_arquivos)
    
    total_imagens = len(nomes_arquivos)
    tamanho_fold = total_imagens // K_FOLDS
    
    print(f"🚀 Iniciando a divisão do K-Fold ({K_FOLDS} Folds)")
    limpar_diretorio(PASTA_KFOLD_BASE)
    
    for k in range(K_FOLDS):
        print(f"🔄 Preparando Fold {k + 1}/{K_FOLDS}...")
        pasta_fold_atual = os.path.join(PASTA_KFOLD_BASE, f"fold_{k+1}")
        
        for split in ['train', 'val']:
            os.makedirs(os.path.join(pasta_fold_atual, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(pasta_fold_atual, split, 'labels'), exist_ok=True)
            
        inicio_val = k * tamanho_fold
        fim_val = (k + 1) * tamanho_fold if k < K_FOLDS - 1 else total_imagens
        
        conjunto_val = nomes_arquivos[inicio_val:fim_val]
        conjunto_treino = [nome for nome in nomes_arquivos if nome not in conjunto_val]
        
        def copiar_dados(conjunto, split):
            for nome in conjunto:
                img_path = Path(PASTA_ORIGEM_IMAGENS) / f"{nome}.jpg"
                if not img_path.exists():
                    img_path = Path(PASTA_ORIGEM_IMAGENS) / f"{nome}.png"
                lbl_path = Path(PASTA_ORIGEM_LABELS) / f"{nome}.txt"
                
                if img_path.exists():
                    shutil.copy(img_path, os.path.join(pasta_fold_atual, split, 'images', img_path.name))
                if lbl_path.exists():
                    shutil.copy(lbl_path, os.path.join(pasta_fold_atual, split, 'labels', lbl_path.name))

        copiar_dados(conjunto_treino, 'train')
        copiar_dados(conjunto_val, 'val')
        
        gerar_yaml_fold_colab(pasta_fold_atual, k + 1)

    print(f"\n✅ Concluído! A pasta '{PASTA_KFOLD_BASE}' está pronta.")
    print("👉 Próximo passo: Comprima a pasta 'dataset_kfold' em um arquivo .zip e faça o upload no Colab.")

if __name__ == "__main__":
    main()