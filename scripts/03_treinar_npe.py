"""Etapa 4 do cronograma: treinamento da NPE (seção 2.5).

Carrega as simulações geradas em 02_simular_treino.py, treina o estimador de
densidade condicional, registra o tempo de treinamento (T_train da seção 2.10) e
salva a posterior em resultados/npe_treinada.pt.
"""

# %% Bibliotecas

import time

import torch
from sbi.inference import NPE_C

from modelo import PASTA_RESULTADOS, PRIOR


# %% Carregar as simulações

# weights_only=False é necessário a partir do torch 2.6, em que o padrão mudou.
simulacoes = torch.load(PASTA_RESULTADOS / "simulacoes_treino.pt", weights_only=False)

theta_train = simulacoes["theta_train"]
x_train = simulacoes["x_train"]

print(f"{len(theta_train)} simulações carregadas.")


# %% Treinamento

print("Inicializando a arquitetura Masked Autoregressive Flow (MAF)...")
inference = NPE_C(prior=PRIOR, density_estimator="maf")
inference = inference.append_simulations(theta_train, x_train)

print("Iniciando o treinamento da rede neural. Isso deve levar alguns minutos...")
inicio = time.time()

# O treinamento separa internamente dados de treino e de validação
density_estimator = inference.train(show_train_summary=True)
posterior_npe = inference.build_posterior(density_estimator)

t_train = time.time() - inicio
print(f"\nTreinamento concluído em {t_train:.2f} segundos.")


# %% Salvar

arquivo = PASTA_RESULTADOS / "npe_treinada.pt"

torch.save(
    {
        "posterior_npe": posterior_npe,
        "t_train": t_train,
    },
    arquivo,
)

print(f"Rede treinada salva em {arquivo}.")
