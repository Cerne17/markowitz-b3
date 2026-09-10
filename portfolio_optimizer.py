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


def pesos_atuais(valores_investidos: list[float]) -> np.ndarray:
    total = sum(valores_investidos)
    if total <= 0:
        raise ValueError("Valor total investido deve ser maior que zero.")
    return np.array([v / total for v in valores_investidos])


def sugestao_rebalanceamento(tickers: list[str], valores_atuais: list[float], pesos_alvo: np.ndarray,
                              aporte: float = 0.0) -> pd.DataFrame:
    """Ajuste sugerido por ativo considerando a carteira atual + um aporte novo a investir agora."""
    total_atual = sum(valores_atuais)
    total_com_aporte = total_atual + aporte
    valores_alvo = pesos_alvo * total_com_aporte
    diffs = valores_alvo - np.array(valores_atuais)
    return pd.DataFrame({
        "ticker": tickers,
        "valor_atual": valores_atuais,
        "peso_atual": np.array(valores_atuais) / total_atual,
        "peso_alvo": pesos_alvo,
        "valor_alvo": valores_alvo,
        "ajuste": diffs,
    })
