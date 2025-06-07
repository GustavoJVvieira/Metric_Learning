# Reconhecimento de Objetos com Metric Learning

O reconhecimento de objetos em imagens é uma tarefa comum em aplicações modernas de Machine Learning. Diversas áreas dependem da análise e comparação de padrões visuais para validar a semelhança entre pessoas, espécies de animais, produtos, entre outros.

## Técnicas Utilizadas

Atualmente, diversas técnicas podem ser utilizadas para identificação de objetos em imagens, como:

- Cálculo de estatísticas de primeira ordem  
- Funções de entropia cruzada  
- Redes Neurais Artificiais, especialmente Redes Convolucionais (CNNs)

## Fluxo de Treinamento e Inferência

Um modelo é treinado em um conjunto de dados, aprendendo padrões nas imagens. Em seguida, ele realiza inferência para reconhecer novos exemplos apresentados. Porém, em muitos casos práticos, pode ser necessário:

- Avaliar registros não presentes no conjunto de treino
- Reconhecer objetos em bases com muitas classes e poucos exemplos por classe

Nesse cenário, o re-treinamento da rede para cada novo exemplo se torna custoso em tempo e recursos.

---

## 1. Metric Learning

Uma abordagem alternativa ao re-treinamento é o **aprendizado por métricas (Metric Learning)**, que busca capturar distâncias entre objetos. Objetos similares têm menor distância entre si.

### Objetivo

Definir uma **função de similaridade** para que casos do mesmo grupo apresentem menor distância em relação a grupos diferentes.

### Exemplo: Reconhecimento Facial

Mesmo com muitas imagens, pode haver poucas por pessoa (classe), dificultando o aprendizado tradicional. Utilizar uma métrica de similaridade pode ser mais eficiente.

![image](https://github.com/user-attachments/assets/9baf5f67-ffbb-4c0d-959b-dec12c5cb77d)


### Métricas Comuns

- **Distância Euclidiana**
- **Distância de Mahalanobis**  
  A distância de Mahalanobis entre duas amostras `xi` e `xj`, dadas por um vetor linear `X = [x1, x2, ..., xn]`, considera uma matriz `M` simétrica e positivamente definida.

- Outras:
  - Matusita
  - Bhattacharyya
  - Kullback-Leibler

---

## 2. Deep Metric Learning

O **Deep Metric Learning** utiliza redes neurais para aprender automaticamente uma métrica de similaridade entre objetos.

### Funcionamento

1. Uma CNN (ou método alternativo como AdaBoost, RankSVM) é treinada com os dados.
2. Uma **métrica de distância** é incorporada como camada da rede.
3. Novos exemplos são comparados com os já indexados na base, com base nessa métrica.

### Framework Visual

![image](https://github.com/user-attachments/assets/e06c1f5a-1360-41f0-b10b-1f2966401a92)


---

## Método de Busca

Após o treinamento, é construído um **índice** com os exemplos da base de dados, ordenado pela métrica de similaridade. A busca por novos exemplos é realizada com algoritmos como o **Nearest Neighbor Search (NNS)**.

### Formulação

Dado um espaço de dimensão `M` e um conjunto de pontos `S`, buscamos encontrar o ponto `q` mais próximo no conjunto `S`.

```math
q* = argmin_{s ∈ S} f(q, s)
```

---

## Estratégias de Busca

![image](https://github.com/user-attachments/assets/51e737a8-f04e-49e9-80dc-a3ce08a6e278)

### Métodos de Busca Exata

- **Busca Linear**: Verifica todos os elementos da base. É simples, porém custosa.
- **Particionamento de Espaço**: Divide o espaço em regiões menores para reduzir o custo da busca.

### Métodos de Aproximação

- **Hierarchical NSW (HNSW)**: Utiliza grafos para representar e buscar aproximações eficientes.

---

## Diferentes Abordagens

Baseado em [Kha Vu, 2020], várias abordagens foram propostas para cálculo de similaridade.

### Contrastive Approach

A ideia principal é **aproximar amostras semelhantes (positivas)** e **afastar amostras diferentes (negativas)**. Utiliza normas de distância como **L2** e aplica uma função de perda no treinamento.

### MoCo (Momentum Contrast)

Utiliza um encoder para gerar representações, comparando uma imagem `q` com um mini-batch de imagens da mesma classe.

### SimCLR

- Aplica uma transformação `t(x)` no exemplo `x`.
- Passa por uma rede `f(.)` e uma projeção `g(.)`.
- O treinamento é feito para **maximizar a concordância** entre representações diferentes do mesmo exemplo.



