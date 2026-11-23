"""Figuras reutilizadas em mais de uma etapa do projeto.

Figuras de uso único ficam no próprio script que as produz.
"""

# %% Bibliotecas

import matplotlib.pyplot as plt

from modelo import ROTULOS_PARAMETROS


# %% Marginais posteriores

def plot_posterior_marginais(amostras, theta_verdadeiro=None, titulo=None):
    """Histogramas das marginais posteriores dos três parâmetros globais.

    amostras: tensor (n_amostras, 3) com log lambda, mu_q e log sigma_q.
    theta_verdadeiro: tensor (3,) com os valores geradores, quando conhecidos.
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i in range(3):
        axes[i].hist(
            amostras[:, i].numpy(),
            bins=30,
            density=True,
            alpha=0.6,
            color="tab:blue",
            label="Posterior",
        )

        if theta_verdadeiro is not None:
            axes[i].axvline(
                theta_verdadeiro[i].item(),
                color="red",
                linestyle="--",
                linewidth=2,
                label="Valor gerador",
            )

        axes[i].set_xlabel(ROTULOS_PARAMETROS[i])
        axes[i].set_ylabel("Densidade")

    axes[0].legend()

    if titulo is not None:
        fig.suptitle(titulo)

    fig.tight_layout()

    return fig, axes
