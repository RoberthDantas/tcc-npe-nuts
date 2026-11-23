"""Etapa 2 do cronograma: exploração visual do modelo gerador.

Simula um conjunto de dados com parâmetros escolhidos e mostra as contagens
observadas ao lado das quantidades latentes que as produziram.
"""

# %% Bibliotecas

import matplotlib.pyplot as plt
import torch

from modelo import simular_modelo


# %% Parâmetros da simulação

# seed = 2027
# torch.manual_seed(seed)

n_arrastos = 30

lambda_ = 200.0
mu_q = -3.0
sigma_q = 0.8


# %% Simulação de um conjunto de dados

n_disponiveis, eta, q, capturas_por_rede, contagens = simular_modelo(
    lambda_=lambda_,
    mu_q=mu_q,
    sigma_q=sigma_q,
    n_arrastos=n_arrastos,
)

arrastos = range(1, n_arrastos + 1)
contagens_plot = contagens.tolist()

contagens


# %% Figura A: contagens observadas

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(arrastos, contagens_plot)
axes[0].set_xlabel("Arrasto")
axes[0].set_ylabel("Contagem combinada")
axes[0].set_title("Captura simulada por arrasto")

axes[1].hist(contagens_plot, bins="auto")
axes[1].set_xlabel("Contagem combinada")
axes[1].set_ylabel("Frequência")
axes[1].set_title("Distribuição das contagens")

fig.tight_layout()
plt.show()

plt.close()


# %% Figura B: quantidades latentes e observadas

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(arrastos, n_disponiveis[:, 0].tolist(), label="Faixa 1")
axes[0].plot(arrastos, n_disponiveis[:, 1].tolist(), label="Faixa 2")
axes[0].set_ylabel("Indivíduos disponíveis")
axes[0].set_title("Indivíduos disponíveis por rede")
axes[0].legend()

axes[1].plot(arrastos, q.tolist(), color="tab:green")
axes[1].axhline(
    torch.sigmoid(torch.tensor(mu_q)).item(),
    color="black",
    linestyle="--",
    label=r"$\operatorname{logit}^{-1}(\mu_q)$",
)
axes[1].set_ylabel(r"$q_i$")
axes[1].set_title("Eficiência de captura")
axes[1].legend()

axes[2].plot(arrastos, capturas_por_rede[:, 0].tolist(), label="Rede 1")
axes[2].plot(arrastos, capturas_por_rede[:, 1].tolist(), label="Rede 2")
axes[2].set_xlabel("Arrasto")
axes[2].set_ylabel("Indivíduos observados")
axes[2].set_title("Indivíduos observados por rede")
axes[2].legend()

fig.tight_layout()
plt.show()

plt.close()
