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


def downside_deviation_anual(retornos_diarios: pd.DataFrame, pesos: np.ndarray, taxa_livre_risco: float) -> float:
    """Volatilidade so da parte ruim (abaixo da taxa livre de risco) - usada no indice de Sortino."""
    rp = retornos_diarios.dot(pesos)
    mar_diario = (1 + taxa_livre_risco) ** (1 / DIAS_UTEIS_ANO) - 1
    downside = np.minimum(rp - mar_diario, 0)
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(DIAS_UTEIS_ANO))


def max_drawdown_portfolio(retornos_diarios: pd.DataFrame, pesos: np.ndarray) -> float:
    """Maior queda historica (pico ao vale) da carteira com esses pesos, buy-and-hold. Numero negativo."""
    rp = retornos_diarios.dot(pesos)
    equity = (1 + rp).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def sortino_ratio(retorno_anual: float, downside_dev_anual: float, taxa_livre_risco: float) -> float:
    return (retorno_anual - taxa_livre_risco) / downside_dev_anual if downside_dev_anual > 0 else 0.0


def calmar_ratio(retorno_anual: float, max_drawdown: float) -> float:
    return retorno_anual / abs(max_drawdown) if max_drawdown < 0 else 0.0


def estatisticas_extras_portfolio(retornos_diarios: pd.DataFrame, pesos: np.ndarray,
                                   retorno_anual: float, taxa_livre_risco: float) -> dict:
    """Sortino, Calmar, CVaR/STARR e max drawdown de uma carteira com pesos dados - complementa
    Sharpe/vol, que so enxergam risco simetrico. Serve p/ qualquer estrategia, nao so as
    otimizadas p/ isso."""
    if not np.any(pesos):
        return {"sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "cvar_anual": 0.0, "starr": 0.0}
    dd_anual = downside_deviation_anual(retornos_diarios, pesos, taxa_livre_risco)
    mdd = max_drawdown_portfolio(retornos_diarios, pesos)
    cvar_anual = cvar_historico_anual(retornos_diarios, pesos)
    return {
        "sortino": sortino_ratio(retorno_anual, dd_anual, taxa_livre_risco),
        "calmar": calmar_ratio(retorno_anual, mdd),
        "max_drawdown": mdd,
        "cvar_anual": cvar_anual,
        "starr": starr_ratio(retorno_anual, cvar_anual, taxa_livre_risco),
    }


def otimiza_max_sortino(media_anual: pd.Series, cov_anual: pd.DataFrame, retornos_diarios: pd.DataFrame,
                         taxa_livre_risco: float, peso_maximo_por_ativo: float = 1.0) -> Portfolio:
    """Maximiza o indice de Sortino (retorno-rf)/downside deviation, em vez de dividir pela vol inteira
    como o Sharpe - penaliza so a variancia "ruim" (abaixo da taxa livre), nao a boa tambem."""
    n = len(media_anual)

    def neg_sortino(pesos):
        retorno = float(np.dot(pesos, media_anual))
        dd = downside_deviation_anual(retornos_diarios, pesos, taxa_livre_risco)
        return -sortino_ratio(retorno, dd, taxa_livre_risco)

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(neg_sortino, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


def otimiza_max_calmar(media_anual: pd.Series, cov_anual: pd.DataFrame, retornos_diarios: pd.DataFrame,
                        taxa_livre_risco: float, peso_maximo_por_ativo: float = 1.0) -> Portfolio:
    """Maximiza o indice de Calmar (retorno anual / maior drawdown historico).

    Drawdown e uma funcao "quebrada" dos pesos (o pior dia pode mudar abruptamente com um
    pequeno ajuste), entao o gradiente numerico do SLSQP e mais ruidoso aqui que em Sharpe/
    Sortino/Vol - pode nao achar o otimo global perfeito, mas converge numa carteira razoavel.
    """
    n = len(media_anual)

    def neg_calmar(pesos):
        retorno = float(np.dot(pesos, media_anual))
        mdd = max_drawdown_portfolio(retornos_diarios, pesos)
        return -calmar_ratio(retorno, mdd)

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(neg_calmar, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


def cvar_historico_anual(retornos_diarios: pd.DataFrame, pesos: np.ndarray, confianca: float = 0.95) -> float:
    """CVaR historico anualizado pela mesma convencao sqrt(tempo) usada na vol/downside deviation
    (aproximacao - CVaR nao escala exatamente assim, mas mantem os indices comparaveis entre si)."""
    rp = retornos_diarios.dot(pesos)
    return var_cvar_historico(rp, confianca)["cvar_diario"] * np.sqrt(DIAS_UTEIS_ANO)


def starr_ratio(retorno_anual: float, cvar_anual: float, taxa_livre_risco: float) -> float:
    return (retorno_anual - taxa_livre_risco) / cvar_anual if cvar_anual > 0 else 0.0


def otimiza_max_starr(media_anual: pd.Series, cov_anual: pd.DataFrame, retornos_diarios: pd.DataFrame,
                       taxa_livre_risco: float, peso_maximo_por_ativo: float = 1.0,
                       confianca: float = 0.95) -> Portfolio:
    """Maximiza o STARR Ratio (retorno-rf)/CVaR - troca a vol inteira do Sharpe pela perda media
    esperada nos piores cenarios (cauda), foco em risco de cauda em vez de dispersao simetrica."""
    n = len(media_anual)

    def neg_starr(pesos):
        retorno = float(np.dot(pesos, media_anual))
        cvar = cvar_historico_anual(retornos_diarios, pesos, confianca)
        return -starr_ratio(retorno, cvar, taxa_livre_risco)

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(neg_starr, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


def otimiza_min_cvar(media_anual: pd.Series, cov_anual: pd.DataFrame, retornos_diarios: pd.DataFrame,
                      taxa_livre_risco: float, peso_maximo_por_ativo: float = 1.0,
                      confianca: float = 0.95) -> Portfolio:
    """Minimiza o CVaR historico (perda media nos piores 5% dos dias), em vez da vol inteira
    como o Min Vol - mais conservador especificamente contra cenarios de cauda ruim."""
    n = len(media_anual)

    def objetivo(pesos):
        return cvar_historico_anual(retornos_diarios, pesos, confianca)

    chute = np.repeat(1 / n, n)
    limites = tuple((0.0, peso_maximo_por_ativo) for _ in range(n))
    resultado = minimize(objetivo, chute, method="SLSQP", bounds=limites, constraints=_restricoes_base(n))
    return desempenho_portfolio(resultado.x, media_anual, cov_anual, taxa_livre_risco)


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


def portfolio_equal_weight(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float) -> Portfolio:
    """Estrategia classica 1/N: mesmo peso p/ todo ativo, sem otimizacao nenhuma.

    Referencia classica da literatura (DeMiguel, Garlappi & Uppal 2009): em varios estudos
    empiricos o 1/N supera carteiras "otimizadas" fora da amostra, por nao depender de
    estimativas de retorno esperado (que sao ruidosas). Bom baseline p/ comparar.
    """
    n = len(media_anual)
    pesos = np.repeat(1 / n, n)
    return desempenho_portfolio(pesos, media_anual, cov_anual, taxa_livre_risco)


def otimiza_risk_parity(media_anual: pd.Series, cov_anual: pd.DataFrame, taxa_livre_risco: float,
                         peso_maximo_por_ativo: float = 1.0) -> Portfolio:
    """Risk Parity (Equal Risk Contribution): pesos tais que cada ativo contribua igualmente
    para a variancia total da carteira, em vez de pesos iguais em R$ (1/N) ou otimizados por
    retorno esperado (Max Sharpe). Nao usa media_anual na otimizacao - so a matriz de
    covariancia -, o que a torna menos sensivel a erro de estimativa de retorno.
    """
    n = len(media_anual)

    def dispersao_contribuicoes(pesos):
        rc = contribuicao_risco(pesos, cov_anual)
        return float(np.sum((rc - 1 / n) ** 2))

    chute = np.repeat(1 / n, n)
    limites = tuple((1e-6, peso_maximo_por_ativo) for _ in range(n))  # >0 p/ contribuicao_risco nao zerar
    resultado = minimize(dispersao_contribuicoes, chute, method="SLSQP", bounds=limites,
                          constraints=_restricoes_base(n))
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
