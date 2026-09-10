"""Teoria Moderna de Portfolio (Markowitz) + indice de Sharpe."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DIAS_UTEIS_ANO = 252


@dataclass
class Portfolio:
    pesos: np.ndarray
    retorno_esperado: float
    volatilidade: float
    sharpe: float


def estatisticas_anuais(retornos_diarios: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    media_anual = retornos_diarios.mean() * DIAS_UTEIS_ANO
    cov_anual = retornos_diarios.cov() * DIAS_UTEIS_ANO
    return media_anual, cov_anual


def desempenho_portfolio(pesos: np.ndarray, media_anual: pd.Series, cov_anual: pd.DataFrame,
                          taxa_livre_risco: float) -> Portfolio:
    retorno = float(np.dot(pesos, media_anual))
    volatilidade = float(np.sqrt(pesos @ cov_anual.values @ pesos))
    sharpe = (retorno - taxa_livre_risco) / volatilidade if volatilidade > 0 else 0.0
    return Portfolio(pesos=pesos, retorno_esperado=retorno, volatilidade=volatilidade, sharpe=sharpe)


def _restricoes_base(n: int):
    return ({"type": "eq", "fun": lambda p: np.sum(p) - 1},)


def otimiza_max_sharpe(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float,
                        peso_maximo_por_ativo: float = 1.0) -> Portfolio:
    n = len(media_anual)

    def neg_sharpe(pesos):
        return -desempenho_portfolio(pesos, media_anual, cov_anual, taxa_livre_risco).sharpe

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(neg_sharpe, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


def otimiza_min_volatilidade(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float,
                              peso_maximo_por_ativo: float = 1.0) -> Portfolio:
    n = len(media_anual)

    def vol(pesos):
        return desempenho_portfolio(pesos, media_anual, cov_anual, taxa_livre_risco).volatilidade

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(vol, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


def fronteira_eficiente(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float,
                         n_pontos: int = 50, peso_maximo_por_ativo: float = 1.0) -> list[Portfolio]:
    n = len(media_anual)
    ret_min = media_anual.min()
    ret_max = media_anual.max()
    alvos = np.linspace(ret_min, ret_max, n_pontos)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    chute = np.repeat(1 / n, n)

    pontos = []
    for alvo in alvos:
        restricoes = (
            {"type": "eq", "fun": lambda p: np.sum(p) - 1},
            {"type": "eq", "fun": lambda p, alvo=alvo: np.dot(p, media_anual) - alvo},
        )

        def vol(pesos):
            return desempenho_portfolio(pesos, media_anual, cov_anual, taxa_livre_risco).volatilidade

        resultado = minimize(vol, chute, method="SLSQP", bounds=limites, constraints=restricoes)
        if resultado.success:
            pontos.append(desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco))
    return pontos


def simula_portfolios_aleatorios(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float,
                                  n_simulacoes: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(media_anual)
    registros = []
    for _ in range(n_simulacoes):
        pesos = rng.random(n)
        pesos /= pesos.sum()
        p = desempenho_portfolio(pesos, media_anual, cov_anual, taxa_livre_risco)
        registros.append((p.retorno_esperado, p.volatilidade, p.sharpe))
    return pd.DataFrame(registros, columns=["retorno", "volatilidade", "sharpe"])


def estatisticas_ativo_individual(precos: pd.Series, taxa_livre_risco: float) -> dict:
    """Retorno/vol/sharpe anualizados + max drawdown de um unico ativo, p/ analise antes de compor a carteira."""
    retornos = precos.pct_change().dropna()
    retorno_anual = float(retornos.mean() * DIAS_UTEIS_ANO)
    vol_anual = float(retornos.std() * np.sqrt(DIAS_UTEIS_ANO))
    sharpe = (retorno_anual - taxa_livre_risco) / vol_anual if vol_anual > 0 else 0.0

    pico = precos.cummax()
    drawdown = precos / pico - 1
    max_drawdown = float(drawdown.min())

    return {
        "retorno_anual": retorno_anual,
        "volatilidade_anual": vol_anual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def contribuicao_risco(pesos: np.ndarray, cov_anual: pd.DataFrame) -> np.ndarray:
    """Fracao da variancia total da carteira explicada por cada ativo (soma = 1).

    Um ativo com peso pequeno mas muito volatil/correlacionado pode dominar o risco
    mesmo sem dominar a alocacao - essa metrica expoe isso, ao contrario do peso sozinho.
    """
    cov = cov_anual.values
    variancia_total = float(pesos @ cov @ pesos)
    if variancia_total <= 0:
        return np.zeros_like(pesos)
    contrib_marginal = cov @ pesos
    return (pesos * contrib_marginal) / variancia_total


def retornos_portfolio(retornos_diarios: pd.DataFrame, pesos: np.ndarray) -> pd.Series:
    return retornos_diarios.dot(pesos)


def var_cvar_historico(retornos_diarios_portfolio: pd.Series, confianca: float = 0.95) -> dict:
    """VaR/CVaR historico (nao-parametrico) diario, a partir da serie real de retornos da carteira."""
    perda_limite = float(np.percentile(retornos_diarios_portfolio, (1 - confianca) * 100))
    cauda = retornos_diarios_portfolio[retornos_diarios_portfolio <= perda_limite]
    cvar = float(cauda.mean()) if len(cauda) > 0 else perda_limite
    return {
        "confianca": confianca,
        "var_diario": -perda_limite,   # perda expressa como numero positivo
        "cvar_diario": -cvar,
    }


LIMITE_ISENCAO_ACOES_MENSAL = 20_000.0


def estima_aliquota_ir(classe: str, total_vendas_acoes_etfs_no_rebalanceamento: float) -> str:
    """Estimativa simplificada da aliquota de IR sobre ganho de capital na venda (regras B3/pessoa fisica).

    Nao calcula o IR em R$ de fato: o app nao rastreia preco medio de compra (custo de
    aquisicao), entao nao da p/ saber o ganho de capital real - so a aliquota/isencao
    que se aplicaria. FIIs nao tem isencao por valor; ETFs tampouco (isencao de R$20k/mes
    e exclusiva de acoes, nao se estende a cotas de fundo por definicao da Receita Federal).
    """
    if classe == "FII":
        return "20% (FIIs nao tem isencao por valor)"
    if classe in ("Acao", "ETF"):
        if classe == "ETF":
            return "15% (ETFs nao tem isencao por valor)"
        if total_vendas_acoes_etfs_no_rebalanceamento <= LIMITE_ISENCAO_ACOES_MENSAL:
            return "Isento (vendas de acoes <= R$20mil/mes)"
        return "15% (vendas de acoes excederam R$20mil no mes)"
    return "15% (estimado - classe nao identificada)"


def pesos_atuais(valores_investidos: list[float]) -> np.ndarray:
    total = sum(valores_investidos)
    if total <= 0:
        raise ValueError("Valor total investido deve ser maior que zero.")
    return np.array([v / total for v in valores_investidos])


def sugestao_rebalanceamento(tickers: list[str], valores_atuais: list[float], pesos_alvo: np.ndarray,
                              aporte: float = 0.0, precos_atuais: list[float] | None = None) -> pd.DataFrame:
    """Ajuste sugerido por ativo considerando a carteira atual + um aporte novo a investir agora.

    Se precos_atuais for informado, tambem calcula quantas cotas/acoes inteiras comprar ou
    vender (arredondado) e o valor real dessa transacao - B3 nao negocia fracao de cota no
    lote padrao, entao o R$ exato do ajuste teorico quase nunca bate com o que da p/ comprar.
    """
    total_atual = sum(valores_atuais)
    total_com_aporte = total_atual + aporte
    valores_alvo = pesos_alvo * total_com_aporte
    diffs = valores_alvo - np.array(valores_atuais)
    peso_atual = np.array(valores_atuais) / total_atual if total_atual > 0 else np.zeros(len(valores_atuais))

    dados = {
        "ticker": tickers,
        "valor_atual": valores_atuais,
        "peso_atual": peso_atual,
        "peso_alvo": pesos_alvo,
        "valor_alvo": valores_alvo,
        "ajuste": diffs,
    }

    if precos_atuais is not None:
        precos_arr = np.array(precos_atuais)
        cotas = np.round(diffs / precos_arr).astype(int)
        dados["preco_atual"] = precos_arr
        dados["cotas_sugeridas"] = cotas
        dados["valor_transacao"] = cotas * precos_arr

    return pd.DataFrame(dados)
