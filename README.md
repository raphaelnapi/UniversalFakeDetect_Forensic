# Extensão Forense — Universal Fake Detect (UFD)

![Imagem de exemplo 1](samples/sample1.png)
![Imagem de exemplo 2](samples/sample2.png)

**Exemplo de uso:**
```bash
python image_score.py --img caminho/para/imagem.jpg --out imagem_saida.png
```

## Escopo e Finalidade

Este repositório é um **fork** do projeto original **Universal Fake Detect (UFD)**.
O código-fonte original e o arquivo `README_UFD.md` descrevem um uso **acadêmico e orientado a benchmarks**,
voltado à avaliação de desempenho de modelos de detecção de imagens sintéticas em conjuntos de dados.

Este documento descreve **script adicional com finalidade forense**, incluído neste fork (image_score.py),
destinado especificamente à **análise de imagem única** em contextos de **perícia digital** com o objetivo
de **estimar a probabilidade** de uma imagem ter sido **gerada por IA**.

> **Importante:**  
> Os scripts forenses aqui descritos realizam **inferência probabilística**.
> Os resultados **não constituem prova conclusiva**, quando considerados de forma isolada.

---

## Objetivo Forense

A extensão forense tem como objetivo:

- Aplicar um **modelo de aprendizado de máquina fixo e bem definido** a uma **imagem individual**
- Produzir um **score numérico** e uma **estimativa probabilística**
- Subsidiar **laudos, pareceres técnicos e análises periciais**
- Preservar **clareza metodológica, reprodutibilidade e auditabilidade**

Esta extensão **não se destina** a:
- validação estatística de modelos
- treinamento ou *fine-tuning*
- comparação de desempenho entre arquiteturas

---

## Visão Metodológica

### Modelo Utilizado
- **Backbone:** CLIP ViT-L/14 (*frozen*)
- **Classificador:** Modelo pré-treinado do Universal Fake Detect (UFD)
- **Tipo de processamento:** Inferência probabilística em imagem única

---

## Scripts Forenses

### `image_score.py`

Script forense para **inferência em imagem única**.

**Funcionalidades:**
- Carrega o modelo UFD/CLIP ViT-L/14 pré-treinado
- Aplica pré-processamento padronizado à imagem
- Calcula um score escalar no intervalo `[0, 1]`
- Apresenta o resultado como **probabilidade estimada de geração por IA** de cada patch 224x224 pixels da imagem
