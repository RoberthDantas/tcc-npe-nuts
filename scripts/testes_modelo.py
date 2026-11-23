"""Testes do modelo gerador (etapa 2 do cronograma).

Verificações simples de que o código em modelo.py reproduz o que está escrito nas
seções 2.2, 2.3 e 2.5 do projeto.

Executar com: python testes_modelo.py
"""

# %% Bibliotecas

import torch

from modelo import (
    PRIOR,
    parametros_de_theta,
    preparar_dados,
    simular_conjuntos,
    simular_modelo,
)


# %% Testes do modelo gerador

def teste_formas():
    """As quantidades simuladas têm as dimensões esperadas."""

    n_arrastos = 30
    n_disponiveis, eta, q, capturas_por_rede, y_total = simular_modelo(
        200.0, -3.0, 0.8, n_arrastos=n_arrastos
    )

    assert n_disponiveis.shape == (n_arrastos, 2)
    assert eta.shape == (n_arrastos,)
    assert q.shape == (n_arrastos,)
    assert capturas_por_rede.shape == (n_arrastos, 2)
    assert y_total.shape == (n_arrastos,)

    print("ok: formas dos tensores simulados")


def teste_captura_nao_excede_disponiveis():
    """Nenhuma rede captura mais indivíduos do que os disponíveis na sua faixa."""

    n_disponiveis, _, _, capturas_por_rede, _ = simular_modelo(
        200.0, -1.0, 0.5, n_arrastos=100
    )

    assert (capturas_por_rede <= n_disponiveis).all()
    assert (capturas_por_rede >= 0).all()

    print("ok: capturas não excedem os indivíduos disponíveis")


def teste_media_das_contagens():
    """A contagem combinada tem média próxima de 2 * lambda * q (seção 2.2)."""

    lambda_ = 500.0
    mu_q = -2.0
    sigma_q = 0.01  # eficiência praticamente constante entre arrastos

    _, _, _, _, y_total = simular_modelo(lambda_, mu_q, sigma_q, n_arrastos=5000)

    esperado = 2 * lambda_ * torch.sigmoid(torch.tensor(mu_q))
    observado = y_total.mean()

    assert torch.isclose(observado, esperado, rtol=0.05)

    print(f"ok: média das contagens {observado:.1f} próxima do esperado {esperado:.1f}")


# %% Testes da preparação dos dados

def teste_preparar_dados():
    """As contagens são ordenadas e transformadas por log(1 + Y) (seção 2.5)."""

    y_total = torch.tensor([7.0, 0.0, 3.0])
    x = preparar_dados(y_total)

    assert torch.allclose(x, torch.log(torch.tensor([1.0, 4.0, 8.0])))
    assert (x.diff() >= 0).all()

    print("ok: contagens ordenadas e transformadas por log(1 + Y)")


# %% Testes do espaço paramétrico

def teste_limites_do_prior():
    """As amostras do prior respeitam os limites da seção 2.3."""

    theta = PRIOR.sample((1000,))

    assert (theta >= PRIOR.base_dist.low).all()
    assert (theta <= PRIOR.base_dist.high).all()

    lambda_min = torch.exp(theta[:, 0]).min()
    lambda_max = torch.exp(theta[:, 0]).max()

    assert lambda_min >= 20.0
    assert lambda_max <= 5000.0

    print(f"ok: prior dentro dos limites (lambda entre {lambda_min:.0f} e {lambda_max:.0f})")


def teste_parametros_de_theta():
    """A conversão de theta para a escala natural inverte as transformações."""

    theta = torch.tensor([torch.log(torch.tensor(200.0)), -3.0, torch.log(torch.tensor(0.8))])
    lambda_, mu_q, sigma_q = parametros_de_theta(theta)

    assert torch.isclose(lambda_, torch.tensor(200.0))
    assert torch.isclose(mu_q, torch.tensor(-3.0))
    assert torch.isclose(sigma_q, torch.tensor(0.8))

    print("ok: conversão de theta para lambda, mu_q e sigma_q")


def teste_simular_conjuntos():
    """A geração de vários conjuntos devolve theta e x com as formas esperadas."""

    theta, x = simular_conjuntos(20, n_arrastos=30)

    assert theta.shape == (20, 3)
    assert x.shape == (20, 30)
    assert (x >= 0).all()

    print("ok: geração de múltiplos conjuntos de dados")


# %% Execução

if __name__ == "__main__":
    torch.manual_seed(42)

    teste_formas()
    teste_captura_nao_excede_disponiveis()
    teste_media_das_contagens()
    teste_preparar_dados()
    teste_limites_do_prior()
    teste_parametros_de_theta()
    teste_simular_conjuntos()

    print("\nTodos os testes passaram.")
