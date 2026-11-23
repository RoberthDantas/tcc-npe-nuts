"""Comparação exploratória entre NUTS e NPE sobre um único conjunto de teste.

Não é a comparação formal da seção 2.9/2.11 (que usa os 300 conjuntos
estratificados de 05_gerar_teste.py, ainda não implementado). Serve como
uma primeira checagem rápida e visual, reaproveitando os resultados já
salvos por 06_inferencia_nuts.py e 07_inferencia_npe.py — por isso não tem
número de etapa no nome, como testes_modelo.py.
"""

# %% Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import torch

from modelo import PASTA_RESULTADOS, ROTULOS_PARAMETROS


# %% Carregar os dois resultados

nuts_resultado = torch.load(PASTA_RESULTADOS / "nuts_teste_unico.pt", weights_only=False)
npe_resultado = torch.load(PASTA_RESULTADOS / "npe_teste_unico.pt", weights_only=False)

theta_true = nuts_resultado["theta_true"]
theta_nuts = nuts_resultado["theta_nuts"]
theta_npe = npe_resultado["theta_npe"]

# Confere que os dois resultados são realmente do mesmo conjunto de teste
assert torch.allclose(theta_true, npe_resultado["theta_true"]), (
    "theta_true difere entre os dois arquivos — NUTS e NPE não foram "
    "aplicados ao mesmo conjunto de teste. Rode 06 e depois 07 de novo."
)


# %% Gráfico lado a lado — marginais de NUTS e NPE sobrepostas

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i in range(3):
    axes[i].hist(
        theta_nuts[:, i].numpy(), bins=30, density=True, alpha=0.5,
        color="tab:orange", label="NUTS",
    )
    axes[i].hist(
        theta_npe[:, i].numpy(), bins=30, density=True, alpha=0.5,
        color="tab:blue", label="NPE",
    )
    axes[i].axvline(
        theta_true[i].item(), color="red", linestyle="--", linewidth=2,
        label="Valor verdadeiro",
    )
    axes[i].set_xlabel(ROTULOS_PARAMETROS[i])
    if i == 0:
        axes[i].legend()

fig.suptitle("NUTS vs NPE — mesmo conjunto de teste")
fig.tight_layout()
plt.show()
plt.close()


# %% Métricas rápidas
# Versão simplificada das métricas da Tabela 1 (seção 2.11), calculada para
# um único conjunto de teste. A versão formal, com todas as métricas (Dμ,
# RW, Wasserstein, cobertura) agregadas sobre os 300 conjuntos
# estratificados, fica para 08_comparar_posteriores.py.

print(f"\n{'Parâmetro':<15}{'Média NUTS':>12}{'Média NPE':>12}{'DP NUTS':>10}{'DP NPE':>10}{'Razão larg.':>14}")
for i, nome in enumerate(["log_lambda", "mu_q", "log_sigma_q"]):
    media_nuts = theta_nuts[:, i].mean().item()
    media_npe = theta_npe[:, i].mean().item()
    dp_nuts = theta_nuts[:, i].std().item()
    dp_npe = theta_npe[:, i].std().item()
    # Razão entre larguras do IC95% (aproximado por 2 * 1.96 * DP), como a
    # métrica R_W da seção 2.11 — valores próximos de 1 indicam incertezas
    # semelhantes entre os dois métodos.
    razao_larguras = (2 * 1.96 * dp_npe) / (2 * 1.96 * dp_nuts)
    print(
        f"{nome:<15}{media_nuts:>12.3f}{media_npe:>12.3f}"
        f"{dp_nuts:>10.3f}{dp_npe:>10.3f}{razao_larguras:>14.3f}"
    )

# Dependência posterior entre log(lambda) e mu_q, nos dois métodos — compara
# se a NPE reproduz a mesma estrutura de confundimento vista no NUTS.
corr_nuts = np.corrcoef(theta_nuts[:, 0].numpy(), theta_nuts[:, 1].numpy())[0, 1]
corr_npe = np.corrcoef(theta_npe[:, 0].numpy(), theta_npe[:, 1].numpy())[0, 1]
print(f"\nCorrelação log(lambda) vs mu_q — NUTS: {corr_nuts:.3f} | NPE: {corr_npe:.3f}")
