"""Etapa 3 do cronograma: simulações de treinamento da NPE.

Gera os pares (theta, dados) usados no treinamento, registra o tempo de simulação
(T_sim da seção 2.10) e salva tudo em resultados/simulacoes_treino.pt.
"""

# %% Bibliotecas

import time

import torch

from modelo import PASTA_RESULTADOS, simular_conjuntos


# %% Configuração

torch.manual_seed(42)

n_conjuntos = 100000
n_arrastos = 30


# %% Geração das simulações

print(f"Gerando {n_conjuntos} simulações de campo...")
inicio = time.time()

theta_train, x_train = simular_conjuntos(n_conjuntos, n_arrastos=n_arrastos)

t_sim = time.time() - inicio
print(f"Simulações concluídas em {t_sim:.2f} segundos.")


# %% Etapa piloto (seção 2.3)

# Verifica se o espaço paramétrico está produzindo muitos conjuntos degenerados
# antes de treinar a rede. x_train está na escala log(1 + Y).
contagens = torch.expm1(x_train)
prop_nulos = (contagens.sum(dim=1) == 0).float().mean()

print(f"Conjuntos com todas as contagens nulas: {prop_nulos:.1%}")
print(f"Contagem máxima observada: {contagens.max():.0f}")


# %% Salvar

arquivo = PASTA_RESULTADOS / "simulacoes_treino.pt"

torch.save(
    {
        "theta_train": theta_train,
        "x_train": x_train,
        "t_sim": t_sim,
        "n_conjuntos": n_conjuntos,
        "n_arrastos": n_arrastos,
    },
    arquivo,
)

print(f"Simulações salvas em {arquivo}.")
