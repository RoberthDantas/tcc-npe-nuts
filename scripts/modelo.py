"""Modelo gerador e espaço paramétrico do projeto (seções 2.2, 2.3 e 2.5).

Este é o único arquivo em que eu defino o modelo gerador e as distribuições
a priori. Todo o resto do pipeline (simulação de treino, treinamento da NPE,
inferência por NUTS) importa as funções daqui — assim garanto que os dois
métodos (NPE e NUTS) estão sempre competindo sobre exatamente o mesmo
problema Bayesiano, sem risco de alguém usar um prior ligeiramente diferente
num script e outro prior noutro.

Esta versão mescla duas coisas:
- a base que já estava validada e rodando (caminho de resultados robusto,
  ordem de ordenação compatível com a rede já treinada);
- alguns ajustes de estilo/limpeza que eu mesmo revisei depois (uso de
  `torch.distributions` como módulo, `log1p`, desempacotamento mais enxuto).
"""

# %% Bibliotecas

from pathlib import Path

import torch
import torch.distributions as tdist
from sbi.utils import BoxUniform


# %% Pasta de resultados
#
# Aqui eu decido ONDE os scripts vão ler/salvar os arquivos .pt (simulações,
# rede treinada, resultados do NUTS etc.). O ponto importante: eu calculo
# esse caminho a partir de `__file__` (a localização deste próprio arquivo
# modelo.py), e NÃO a partir do diretório de trabalho (cwd) de quem está
# rodando o notebook.
#
# Por quê isso importa: se eu usasse só `Path("resultados")`, o Python
# procuraria/criaria essa pasta relativa a "de onde o notebook foi aberto" —
# e isso muda dependendo de como o VS Code/Jupyter está configurado no
# momento. Já tive bugs assim antes (ModuleNotFoundError, NameError) só
# porque o diretório de trabalho não era o que eu esperava. Usando
# `__file__`, o caminho fica fixo e sempre aponta para a mesma pasta
# `resultados/` (um nível acima da pasta onde este modelo.py está), não
# importa de onde eu rode o código.
PASTA_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)


# %% Espaço paramétrico (seção 2.3)
#
# theta = (log(lambda), mu_q, log(sigma_q)) é o vetor de parâmetros globais
# que tanto a NPE quanto o NUTS tentam recuperar. Eu trabalho na escala log
# para lambda e sigma_q porque essas duas variáveis só fazem sentido como
# valores positivos — rodar a inferência na escala log evita que o
# amostrador (ou a rede) perca tempo/densidade em regiões impossíveis
# (lambda negativo, por exemplo).

NOMES_PARAMETROS = ["log_lambda", "mu_q", "log_sigma_q"]
ROTULOS_PARAMETROS = [r"$\log(\lambda)$", r"$\mu_q$", r"$\log(\sigma_q)$"]

# Limites do prior, direto da seção 2.3 do projeto:
#   log(lambda)   ~ Uniforme[log(20), log(5000)]   -> abundância latente
#   mu_q          ~ Uniforme[-6, 0]                 -> eficiência média (logit)
#   log(sigma_q)  ~ Uniforme[log(0.2), log(2.5)]    -> variabilidade da eficiência
#
# Uso o BoxUniform do pacote sbi porque ele serve pros dois propósitos ao
# mesmo tempo: é o objeto que eu amostro (`PRIOR.sample()`) para gerar os
# dados de treino/teste, e é também o prior que eu passo pro NPE_C — assim
# não corro o risco de "esquecer" de manter os dois sincronizados.
PRIOR = BoxUniform(
    low=torch.tensor([torch.log(torch.tensor(20.0)), -6.0, torch.log(torch.tensor(0.2))]),
    high=torch.tensor([torch.log(torch.tensor(5000.0)), 0.0, torch.log(torch.tensor(2.5))]),
)


def parametros_de_theta(theta):
    """Converto theta = (log lambda, mu_q, log sigma_q) para a escala natural.

    Isso é só desfazer a transformação log: a rede e o NUTS trabalham em
    escala log, mas o modelo gerador (Poisson, Normal, Binomial) precisa
    dos valores na escala real (lambda > 0, sigma_q > 0).
    """

    lambda_ = torch.exp(theta[0])
    mu_q = theta[1]
    sigma_q = torch.exp(theta[2])

    return lambda_, mu_q, sigma_q


# %% Modelo gerador (seção 2.2)

def simular_modelo(lambda_, mu_q, sigma_q, n_arrastos=30):
    """Simulo um "dia de pesca" completo: quantos peixes existem, quão
    eficiente a rede é em cada arrasto, e quantos peixes acabam sendo
    efetivamente capturados.

    Devolvo as variáveis latentes (que na vida real eu nunca observo) junto
    com a contagem final, porque em alguns pontos do projeto (como gerar o
    dado de teste "verdadeiro" para validar NUTS/NPE) é útil inspecionar
    essas variáveis latentes também.
    """

    # Indivíduos disponíveis nas duas faixas percorridas por cada arrasto.
    # Isso é o processo biológico latente: eu nunca observo esse número
    # diretamente, só a parte que a rede efetivamente captura.
    n_disponiveis = tdist.Poisson(lambda_).sample((n_arrastos, 2))

    # Eficiência de captura do arrasto (compartilhada pelas duas redes do
    # mesmo arrasto). Trabalho na escala logito (eta) e depois aplico
    # sigmoid para trazer de volta pra [0, 1], que é a escala de uma
    # probabilidade de captura.
    eta = tdist.Normal(mu_q, sigma_q).sample((n_arrastos,))
    q = torch.sigmoid(eta)

    # Dado quantos peixes estavam disponíveis e qual a eficiência da rede,
    # simulo quantos peixes cada rede efetivamente capturou (Binomial).
    capturas_por_rede = tdist.Binomial(
        total_count=n_disponiveis,
        probs=q[:, None],
    ).sample()

    # A única coisa que eu realmente "anoto na prancheta" em campo: a soma
    # das duas redes por arrasto. É isso que entra no modelo de inferência,
    # nunca as variáveis latentes acima.
    y_total = capturas_por_rede.sum(dim=1)

    return n_disponiveis, eta, q, capturas_por_rede, y_total


# %% Preparação dos dados para a rede (seção 2.5)

def preparar_dados(y_total):
    """Transformo as contagens observadas no formato que entra na NPE.

    Duas transformações, nessa ordem exata:
    1. Ordeno as contagens (crescente). Os arrastos não têm uma ordem
       temporal relevante para o problema, então a ordem bruta em que eles
       aparecem no vetor não deveria importar para a rede — ordenar remove
       essa variação artificial e deixa o problema mais fácil de aprender.
    2. Aplico log(1 + Y) para comprimir a escala (contagens podem variar de
       0 a milhares) e estabilizar a variância.

    Importante: a rede NPE que eu já treinei aprendeu com dados ordenados de
    forma CRESCENTE. Se eu mudar essa ordem aqui (por exemplo, para
    decrescente) sem retreinar a rede do zero, a inferência continua
    rodando sem erro nenhum, mas o resultado fica sutilmente errado, porque
    a rede nunca viu dados nesse formato durante o treino. Por isso não
    mexo nessa ordem sem motivo forte — e se mudar, preciso lembrar de
    rodar 02 e 03 de novo (regerar simulações e retreinar).
    """

    y_ordenado, _ = torch.sort(y_total)

    # log1p(y) é equivalente a log(1 + y), só que numericamente mais estável
    # para valores de y próximos de zero.
    return torch.log1p(y_ordenado)


# %% Geração de vários conjuntos de dados

def simular_conjuntos(n_conjuntos, n_arrastos=30):
    """Repito o processo "sortear theta do prior -> simular um conjunto de
    dados" `n_conjuntos` vezes seguidas. É essa função que uso tanto para
    gerar o conjunto de treino da NPE (M=100.000 no pipeline atual) quanto,
    no futuro, para gerar os conjuntos de teste estratificados (seção 2.6).

    Devolvo:
    - theta_train, com formato (n_conjuntos, 3) — os parâmetros verdadeiros
      usados em cada simulação (o que a rede deve aprender a recuperar);
    - x_train, com formato (n_conjuntos, n_arrastos) — os dados já
      pré-processados (ordenados + log1p) que a rede efetivamente vê.

    As variáveis latentes (n_disponiveis, eta, q, capturas_por_rede) são
    descartadas aqui de propósito: a NPE nunca deve ter acesso a elas, só às
    contagens finais, exatamente como aconteceria num cenário real de
    campo.
    """

    lista_theta, lista_x = [], []

    for _ in range(n_conjuntos):
        theta = PRIOR.sample()
        lambda_, mu_q, sigma_q = parametros_de_theta(theta)

        # Descarto tudo que não for y_total (variáveis latentes) com o `*_,`.
        *_, y_total = simular_modelo(lambda_, mu_q, sigma_q, n_arrastos=n_arrastos)

        lista_theta.append(theta)
        lista_x.append(preparar_dados(y_total))

    return torch.stack(lista_theta), torch.stack(lista_x)