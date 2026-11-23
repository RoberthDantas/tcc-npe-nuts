# Inferência Bayesiana amortizada (NPE) × NUTS em um modelo ecológico de captura

Neste repositório reúno o código, os resultados e os registros do meu Trabalho de Conclusão de Curso, **"Acurácia e eficiência computacional da inferência Bayesiana amortizada em um modelo ecológico com variáveis latentes"**, no Bacharelado Interdisciplinar em Ciência e Tecnologia do Mar (Instituto do Mar, Unifesp – Campus Baixada Santista, 2026).

No trabalho, comparo a Estimação Neural da Distribuição Posterior (NPE, pacote [`sbi`](https://github.com/sbi-dev/sbi)) com o MCMC pelo No-U-Turn Sampler (NUTS, [Pyro](https://pyro.ai)). Aplico os dois métodos aos mesmos conjuntos de dados simulados, com o mesmo modelo, o mesmo espaço paramétrico e as mesmas distribuições a priori. Com isso, meço o quanto a NPE reproduz as posteriores do NUTS e a partir de quantos conjuntos analisados o custo inicial dela (simular e treinar) se paga.

> **Situação:** o pipeline está completo e os resultados finais já foram gerados. O texto do TCC está em revisão com o meu orientador.

## Modelo

Cada conjunto de dados representa uma espécie capturada como fauna acompanhante na pesca de arrasto de duas portas, com $n = 30$ arrastos. Em cada arrasto $i$, duas redes $r = 1, 2$ percorrem faixas vizinhas do fundo:

$$N_{ir} \sim \text{Poisson}(\lambda), \qquad \eta_i \sim \text{Normal}(\mu_q, \sigma_q^2), \qquad q_i = \text{logit}^{-1}(\eta_i)$$

$$Y_{ir} \mid N_{ir}, q_i \sim \text{Binomial}(N_{ir}, q_i), \qquad Y_{i+} = Y_{i1} + Y_{i2}$$

Só a soma $Y_{i+}$ é observada. A abundância $\lambda$ e a eficiência de captura $q_i$ não são observadas diretamente, e combinações diferentes das duas geram contagens parecidas. Os parâmetros que infiro são $\theta = (\log\lambda,\ \mu_q,\ \log\sigma_q)$, com priors uniformes:

| Parâmetro | Prior |
| --- | --- |
| $\log\lambda$ | $U[\log 20,\ \log 5000)$ |
| $\mu_q$ | $U[-6,\ 0)$ |
| $\log\sigma_q$ | $U[\log 0{,}2,\ \log 2{,}5)$ |

- **NUTS:** uso a forma marginalizada $Y_{i+} \mid q_i \sim \text{Poisson}(2\lambda q_i)$, com $\eta_i = \mu_q + \sigma_q z_i$ (parametrização não centrada).
- **NPE:** a rede recebe as 30 contagens ordenadas em ordem crescente e transformadas por $\log(1 + y)$, e aprende $q_\phi(\theta \mid Y) \approx p(\theta \mid Y)$ com um fluxo normalizante (MAF, `NPE_C` do `sbi`).

## Principais resultados

**Calibração da NPE (SBC, 1.000 conjuntos):** a cobertura dos intervalos de 95% ficou entre 0,926 e 0,936, com histogramas de rank em leve forma de U. Ou seja, os intervalos da NPE saíram um pouco estreitos demais, sobretudo em $\log\sigma_q$.

**Qualidade do NUTS como referência:** rodei o NUTS em 60 dos 300 conjuntos de teste (20 por faixa de $\mu_q$) e refiz com passo menor os 19 que tiveram divergências. Depois disso, 56 dos 60 passam no critério moderado (R̂ ≤ 1,05, ESS ≥ 100, divergências ≤ 1%) e 35 no estrito (R̂ ≤ 1,01, ESS ≥ 400). Os quatro que não passam são todos da faixa $\mu_q \in [-2, 0)$.

**Concordância NPE × NUTS (critério moderado, medianas por faixa e parâmetro):**

| Métrica | Valor | Leitura |
| --- | --- | --- |
| $\lvert D_\mu \rvert$ | 0,05 a 0,09 | diferença de médias em desvios-padrão do NUTS |
| $R_W$ | 0,97 a 1,00 | razão entre as larguras dos intervalos de 95% |
| Correlação $\log\lambda \times \mu_q$ | −0,96 a −0,99 nos dois métodos | a NPE reproduz a dependência entre abundância e eficiência |

**Custo computacional:**

| Item | Tempo |
| --- | --- |
| Simulação de 100.000 conjuntos de treino ($T_{sim}$) | 39,8 s |
| Treino da NPE ($T_{train}$) | 969,4 s (16,2 min) |
| NPE por conjunto ($t_{NPE}$, mediana) | 0,056 s |
| NUTS por conjunto ($t_{NUTS}$, máquina livre) | 62,0 min |
| Ponto de amortização $R^* = (T_{sim} + T_{train}) / (t_{NUTS} - t_{NPE})$ | **0,27 conjunto** |
| 300 conjuntos | ~310 h no NUTS × ~17 min na NPE (≈ 1.088 vezes) |

Como $R^* < 1$, o custo inicial da NPE já fica abaixo do de uma única inferência por NUTS.

**Sensibilidade do treino (`11_estudo_treino.py`):** com 10.000 simulações, a mediana de $\lvert D_\mu \rvert$ foi 0,26. Com 100.000 caiu para cerca de 0,07, e com 200.000 ficou em 0,06. Treinando com cinco sementes e as mesmas 100.000 simulações, obtive medianas entre 0,065 e 0,115.

## Estrutura do repositório

```
.
├── scripts/                 # pipeline numerado (no meu computador, a pasta se chama teste/)
│   ├── modelo.py            # simulador, priors, preparar_dados e PASTA_RESULTADOS
│   ├── graficos.py          # figuras reutilizadas
│   ├── testes_modelo.py     # testes automáticos do simulador
│   ├── 01_explorar_modelo.py ... 11_estudo_treino.py
├── resultados/              # saídas de cada etapa (ver resultados/README.md)
├── legado/                  # protótipos que não fazem parte do pipeline final
├── requirements.txt
└── CITATION.cff
```

Os scripts gravam e leem tudo em `resultados/`, uma pasta acima de `scripts/`. O caminho é calculado em `modelo.py` a partir do próprio arquivo, então funciona de qualquer diretório.

## Como reproduzir

### Ambiente

```bash
python -m venv .venv            # Python 3.11
.venv\Scripts\activate          # Windows (no Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cd scripts
python testes_modelo.py         # confere o simulador
python 02_simular_treino.py     # e assim por diante, na ordem abaixo
```

No Windows, rode cada script como arquivo, no terminal ou pelo botão "Executar arquivo Python" do VS Code, e não na janela interativa. O `08` e o `08b` usam vários processos, que travam quando iniciados de um notebook ou da janela interativa.

### Ordem de execução

| Etapa | Script | Saída em `resultados/` | Tempo aproximado |
| --- | --- | --- | --- |
| Testes automáticos do simulador | `testes_modelo.py` | — | segundos |
| Explorar o modelo (figuras de um conjunto simulado) | `01_explorar_modelo.py` | — (figuras na tela) | segundos |
| Simular 100.000 conjuntos de treino (semente 42) e etapa piloto | `02_simular_treino.py` | `simulacoes_treino.pt` | 40 s |
| Treinar a NPE (`NPE_C`, MAF) | `03_treinar_npe.py` | `npe_treinada.pt` | 16 min |
| Calibração por SBC (1.000 conjuntos, semente 7) | `04_calibracao_sbc.py` | `sbc_npe.png`, `sbc_npe.pt` | minutos |
| 300 conjuntos de teste, 100 por faixa de $\mu_q$ (semente 2027) | `05_gerar_teste.py` | `conjuntos_teste.pt` | segundos |
| NUTS em um conjunto (validação inicial) | `06_inferencia_nuts.py` | `nuts_teste_unico.pt` | ~1 h |
| NPE no mesmo conjunto | `07_inferencia_npe.py` | `npe_teste_unico.pt` | segundos |
| NPE nos 300 conjuntos | `07_inferencia_npe_300.py` | `npe_300.pt` | 22 s |
| NUTS em 60 conjuntos (20 por faixa), com checkpoint | `08_nuts_300.py` | `nuts_lote/`, `nuts_300.pt` | dezenas de horas |
| Refazer os 19 conjuntos com divergências (`target_accept_prob` = 0,95, aquecimento 1.500) | `08b_refazer_problematicos.py` | `nuts_lote_refeito/` | ~27 h |
| Comparação NPE × NUTS (critérios estrito e moderado) | `09_comparar_posteriores.py` | `metricas_posteriores.csv`, `resumo_metricas_por_faixa.csv`, `comparacao_metricas_*.png` | segundos |
| Custos e ponto de amortização $R^*$ | `10_custos_amortizacao.py` | `custo_acumulado.png` | segundos |
| Sensibilidade do treino (orçamento de simulação e sementes) | `11_estudo_treino.py` | `estudo_treino/` | ~3 h 30 |

Deixei em `resultados/` as saídas de todas as etapas. Para refazer só a comparação e os custos, basta rodar o `09` e o `10`, sem repetir as dezenas de horas de NUTS. O `08` e o `11` retomam de onde pararam, porque cada conjunto ou treino é salvo assim que termina.

### Cuidados

- **Ordem das contagens:** treinei a NPE com as contagens em ordem **crescente** (`preparar_dados` em `modelo.py`). Mudar essa ordem não dá erro, mas deixa os resultados errados, e exige refazer o `02` e o `03`.
- **Sementes:** cada etapa usa uma semente fixa, indicada no cabeçalho do script. As de treino (42), SBC (7), teste (2027) e das 200.000 simulações do estudo de treino (2028) são independentes entre si.
- **Versão do PyTorch:** a mesma semente reproduz exatamente os parâmetros sorteados, mas as contagens simuladas podem mudar um pouco entre versões do PyTorch. Rodei o `02` com outra versão: os 100.000 θ saíram idênticos e 97% das contagens coincidiram com as de `simulacoes_treino.pt`. Para refazer tudo bit a bit, use a mesma versão do PyTorch que eu usei.
- **Cadeias do NUTS em sequência:** o `num_chains > 1` do Pyro trava no Windows. Por isso, as 4 cadeias de cada conjunto rodam uma após a outra, e o paralelismo do `08` é entre conjuntos (3 processos de uma thread).

## Ambiente computacional dos resultados

Gerei todos os resultados no meu computador:

| Item | Configuração |
| --- | --- |
| Sistema | Windows, VS Code |
| Python | 3.11.16 (ambiente conda) |
| Processador | AMD Ryzen 7 4800HS, 2,9 GHz, 8 núcleos |
| Memória | 8 GB |
| GPU | Radeon Graphics integrada (não usada; tudo rodou em CPU) |

Os tempos são de relógio (wall-clock). No `08`, rodei o NUTS com 3 conjuntos ao mesmo tempo (mediana de 59,9 min por conjunto). Para o $R^*$, usei os 62,0 min de um conjunto rodado sozinho (`06`).

## Referências

- BINGHAM, E. et al. Pyro: Deep Universal Probabilistic Programming. *Journal of Machine Learning Research*, v. 20, n. 28, p. 1–6, 2019.
- CRANMER, K.; BREHMER, J.; LOUPPE, G. The frontier of simulation-based inference. *PNAS*, v. 117, n. 48, p. 30055–30062, 2020. https://doi.org/10.1073/pnas.1912789117
- HOFFMAN, M. D.; GELMAN, A. The No-U-Turn Sampler. *Journal of Machine Learning Research*, v. 15, n. 47, p. 1593–1623, 2014.
- TALTS, S. et al. Validating Bayesian Inference Algorithms with Simulation-Based Calibration. arXiv:1804.06788, 2018.
- TEJERO-CANTERO, Á. et al. sbi: A toolkit for simulation-based inference. *Journal of Open Source Software*, v. 5, n. 52, p. 2505, 2020. https://doi.org/10.21105/joss.02505
- TRENKEL, V. M.; SKAUG, H. J. Disentangling the effects of capture efficiency and population abundance on catch data using random effects models. *ICES Journal of Marine Science*, v. 62, n. 8, p. 1543–1555, 2005. https://doi.org/10.1016/j.icesjms.2005.05.010

## Autoria e uso

Sou o autor deste trabalho, Roberth Dantas Gomes. Os arquivos `modelo.py`, `graficos.py`, `testes_modelo.py` e as etapas `01` a `03` partem do código de referência que meu orientador me forneceu; adaptei o `modelo.py` e o `graficos.py` para este pipeline.

Ainda não defini uma licença: o conteúdo pode ser consultado, mas o reuso depende da minha autorização. Para citar, use o arquivo `CITATION.cff` (botão "Cite this repository" do GitHub).
