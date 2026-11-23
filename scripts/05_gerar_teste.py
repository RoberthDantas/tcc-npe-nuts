"""Etapa 5 do cronograma: conjuntos de teste estratificados (seção 2.6).

Gera os 300 conjuntos de teste usados na comparação final entre NUTS e NPE:
100 conjuntos em cada faixa de mu_q, [-6,-4), [-4,-2) e [-2,0), com log(lambda)
e log(sigma_q) sorteados do prior de referência (modelo.PRIOR) e mu_q sorteado
de forma uniforme dentro da faixa. Os conjuntos são independentes dos de
treino: usam uma semente diferente da do 02_simular_treino.py (42), e o
script confere que nenhum theta de teste coincide com um theta de treino.

Guarda tanto as contagens brutas (y_test, entrada do NUTS) quanto as contagens
já preparadas (x_test, entrada da NPE), para que os dois métodos sejam
aplicados exatamente aos mesmos dados.
"""

# %% Bibliotecas

import torch

from modelo import (
    PASTA_RESULTADOS,
    PRIOR,
    parametros_de_theta,
    preparar_dados,
    simular_modelo,
)


# %% Configuração

# Semente diferente da usada no treino (42), para garantir independência.
SEMENTE_TESTE = 2027

n_arrastos = 30
n_por_faixa = 100
FAIXAS_MU_Q = [(-6.0, -4.0), (-4.0, -2.0), (-2.0, 0.0)]

torch.manual_seed(SEMENTE_TESTE)


# %% Geração dos conjuntos de teste

lista_theta, lista_y, lista_faixa = [], [], []

for indice_faixa, (mu_min, mu_max) in enumerate(FAIXAS_MU_Q):
    for _ in range(n_por_faixa):
        # log(lambda) e log(sigma_q) vêm do prior de referência...
        theta = PRIOR.sample()
        # ...e mu_q é sorteado uniformemente dentro da faixa [mu_min, mu_max).
        theta[1] = mu_min + (mu_max - mu_min) * torch.rand(())

        lambda_, mu_q, sigma_q = parametros_de_theta(theta)
        _, _, _, _, y_total = simular_modelo(lambda_, mu_q, sigma_q, n_arrastos=n_arrastos)

        lista_theta.append(theta)
        lista_y.append(y_total)
        lista_faixa.append(indice_faixa)

theta_test = torch.stack(lista_theta)
y_test = torch.stack(lista_y)
x_test = torch.stack([preparar_dados(y) for y in y_test])
faixa = torch.tensor(lista_faixa)

print(f"{len(theta_test)} conjuntos de teste gerados "
      f"({n_por_faixa} por faixa de mu_q).")


# %% Checagem piloto por faixa
# Quanto menor mu_q, menor a eficiência de captura e mais contagens baixas
# (ou totalmente nulas) devem aparecer — esperado, não é erro.

for indice_faixa, (mu_min, mu_max) in enumerate(FAIXAS_MU_Q):
    y_faixa = y_test[faixa == indice_faixa]
    prop_nulos = (y_faixa.sum(dim=1) == 0).float().mean()
    print(f"mu_q em [{mu_min:.0f}, {mu_max:.0f}): "
          f"conjuntos totalmente nulos = {prop_nulos:.1%}, "
          f"contagem máxima = {y_faixa.max():.0f}, "
          f"contagem total mediana = {y_faixa.sum(dim=1).median():.0f}")


# %% Independência em relação ao conjunto de treino

arquivo_treino = PASTA_RESULTADOS / "simulacoes_treino.pt"

if arquivo_treino.exists():
    simulacoes = torch.load(arquivo_treino, weights_only=False)
    distancia_minima = torch.cdist(theta_test, simulacoes["theta_train"]).min().item()
    print(f"Menor distância entre um theta de teste e um theta de treino: {distancia_minima:.4f}")
    assert distancia_minima > 0, (
        "Algum theta de teste coincide com um theta de treino — "
        "troque SEMENTE_TESTE e gere de novo."
    )
else:
    print("simulacoes_treino.pt não encontrado: checagem de independência pulada.")


# %% Salvar

arquivo = PASTA_RESULTADOS / "conjuntos_teste.pt"

torch.save(
    {
        "theta_test": theta_test,
        "y_test": y_test,
        "x_test": x_test,
        "faixa": faixa,
        "faixas_mu_q": FAIXAS_MU_Q,
        "semente": SEMENTE_TESTE,
        "n_arrastos": n_arrastos,
        "n_por_faixa": n_por_faixa,
    },
    arquivo,
)

print(f"Conjuntos de teste salvos em {arquivo}.")
