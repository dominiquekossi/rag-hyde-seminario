# RAG com HyDE: baseline, teste em português e app web

Trabalho em trio de Seminários de Computação I (ECO0061), UFC Sobral. Este repositório tem a parte de Kossi Houessou: os testes do HyDE, o teste em português com o Quati e o app web.

Artigo base: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), [arXiv 2212.10496](https://arxiv.org/abs/2212.10496).

## O que é o HyDE

Na busca densa normal, a pergunta vira um vetor e o sistema procura os documentos com vetor parecido. No HyDE, um LLM primeiro escreve passagens hipotéticas que responderiam à pergunta. O vetor final da consulta é a média entre o vetor da pergunta e os vetores das passagens. A busca depois é a mesma. As passagens podem ter erros, mas usam o vocabulário do assunto, e isso ajuda a achar os documentos certos.

## O que há aqui

| Pasta | Conteúdo |
| --- | --- |
| `notebooks/Parte3_Baseline_HyDE_SciFact.ipynb` | BM25, busca densa e HyDE na SciFact, com teste de significância e ablações |
| `notebooks/Parte3b_Quati_HyDE.ipynb` | O mesmo teste em português, em um subconjunto do Quati (com as saídas da nossa rodada) |
| `notebooks/Parte1_App_RAG_HyDE.ipynb` | App web de demonstração (Gradio), pronto para rodar no Colab |
| `resultados/` | Tabelas dos testes em CSV (ver `LEIAME.txt` na pasta) |
| `app/` | Código do app: `rag_core.py` (busca, HyDE e resposta) e `app.py` (interface) |

## Como rodar

Abra o notebook no Colab, escolha **Ambiente de execução > Alterar o tipo de ambiente de execução > GPU T4** e rode as células em ordem. Sem GPU, a geração das passagens fica muito lenta (mais de 300 s por busca no app, contra cerca de 8 s na T4).

| Notebook | Abrir no Colab |
| --- | --- |
| Baseline SciFact | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dominiquekossi/rag-hyde-seminario/blob/main/notebooks/Parte3_Baseline_HyDE_SciFact.ipynb) |
| Quati (português) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dominiquekossi/rag-hyde-seminario/blob/main/notebooks/Parte3b_Quati_HyDE.ipynb) |
| App web | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dominiquekossi/rag-hyde-seminario/blob/main/notebooks/Parte1_App_RAG_HyDE.ipynb) |

O notebook do app cria os arquivos do `app/` sozinho e gera um link público (gradio.live) que dura no máximo uma semana e some quando a sessão termina.

## Configuração dos testes

- Encoders: `facebook/contriever` (só na SciFact) e `intfloat/multilingual-e5-small`.
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`, 4 passagens por consulta, até 200 tokens, temperatura 0,7, semente 42.
- Métricas: nDCG@10 e Recall@100. Diferenças pareadas por consulta, com intervalo de confiança por bootstrap (10.000 reamostragens), teste t pareado e teste de Wilcoxon.

## Resultados

**SciFact** (300 consultas, em inglês):

| Método | nDCG@10 | Recall@100 |
| --- | --- | --- |
| BM25 | 0,652 | 0,873 |
| Denso (Contriever) | 0,649 | 0,926 |
| Denso (mE5-small) | 0,677 | 0,925 |
| HyDE (Contriever) | 0,677 | 0,955 |
| HyDE (mE5-small) | 0,716 | 0,938 |

Com o mE5, o HyDE melhora o nDCG@10 em 0,039 e a diferença é significativa. Com o Contriever, a melhora é de 0,028 e não é significativa a 5%. Nas ablações, o melhor resultado com o mE5 foi com 4 passagens.

**Quati** (português, 49 consultas, subconjunto de 21.843 passagens, mE5-small):

| Método | nDCG@10 | Recall@100 |
| --- | --- | --- |
| BM25 | 0,426 | 0,727 |
| Denso | 0,528 | 0,903 |
| HyDE com prompt em português | 0,573 | 0,924 |
| HyDE com prompt em inglês | 0,535 | 0,931 |

O prompt em português ganhou do prompt em inglês, e o app usa o prompt em português. O ganho do HyDE sobre a busca densa (+0,045) é significativo a 5%, mas o intervalo de confiança quase chega a zero.

## Limitações

- O Quati foi testado em um subconjunto (todas as passagens com nota mais 20.000 sorteadas), então os números não são comparáveis com os do artigo do Quati. As notas de relevância foram dadas por um LLM.
- São poucas consultas no Quati (49), e há uma única geração de passagens por base. Outra semente pode mudar os números.
- Medimos só a etapa de busca. Não avaliamos a qualidade das respostas geradas pelo app.
- Os p-valores não têm correção para comparações múltiplas.
