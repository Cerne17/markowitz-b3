"""Busca precos historicos de ativos da B3 via Yahoo Finance."""
import datetime as dt
import json

import pandas as pd
import yfinance as yf


def normaliza_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker.endswith(".SA"):
        ticker += ".SA"
    return ticker


def baixar_precos(tickers: list[str], data_inicio: str, data_fim: str | None = None) -> pd.DataFrame:
    """Retorna DataFrame de precos de fechamento ajustado, colunas = tickers."""
    tickers = [normaliza_ticker(t) for t in tickers]
    data_fim = data_fim or dt.date.today().isoformat()

    dados = yf.download(
        tickers,
        start=data_inicio,
        end=data_fim,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if dados.empty:
        raise ValueError("Nenhum dado retornado pelo Yahoo Finance. Verifique os tickers e o periodo.")

    precos = pd.DataFrame({t: dados[t]["Close"] for t in tickers if t in dados.columns.get_level_values(0)})

    faltando = [t for t in tickers if t not in precos.columns]
    if faltando:
        raise ValueError(f"Tickers sem dados (verifique o codigo): {', '.join(faltando)}")

    precos = precos.dropna(how="all").ffill().dropna()
    if precos.empty:
        raise ValueError("Series de precos vazias apos limpeza. Verifique tickers e periodo.")
    return precos


def calcula_retornos_diarios(precos: pd.DataFrame) -> pd.DataFrame:
    return precos.pct_change().dropna()


def carrega_universo_ativos(caminho: str = "ativos_b3.json") -> list[dict]:
    """Amostra curada de ativos B3 por setor, p/ busca/selecao na UI."""
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados["ativos"]


def obter_info_ativo(ticker: str) -> dict:
    """Info fundamentalista best-effort (Yahoo Finance pode falhar/atrasar/nao ter o dado - nao é critico).

    market_cap ("TAM") e price_to_book (P/VP) fazem sentido p/ acoes e FIIs; capex so existe p/ acoes
    (FIIs nao reportam capex tradicional) e ETFs nao tem nenhum desses (sao None).
    """
    ticker = normaliza_ticker(ticker)
    t = yf.Ticker(ticker)
    resultado = {}

    try:
        info = t.info
        resultado.update({
            "nome_longo": info.get("longName") or info.get("shortName") or ticker,
            "setor": info.get("sector"),
            "industria": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "price_to_book": info.get("priceToBook"),
            "moeda": info.get("currency"),
            "maxima_52_sem": info.get("fiftyTwoWeekHigh"),
            "minima_52_sem": info.get("fiftyTwoWeekLow"),
        })
    except Exception:
        pass

    try:
        cf = t.cashflow
        if cf is not None and "Capital Expenditure" in cf.index:
            capex = cf.loc["Capital Expenditure"].dropna()
            if not capex.empty:
                resultado["capex_recente"] = float(capex.iloc[0])
                data_capex = capex.index[0]
                resultado["capex_data"] = data_capex.date().isoformat() if hasattr(data_capex, "date") \
                    else str(data_capex)
    except Exception:
        pass

    return resultado


HORIZONTES_EXPLORAR = {
    "6 meses": 182,
    "1 ano": 365,
    "2 anos": 730,
    "3 anos": 1095,
    "5 anos": 1825,
    "10 anos": 3650,
    "Maximo": None,
}


def data_inicio_por_horizonte(rotulo: str) -> str:
    dias = HORIZONTES_EXPLORAR.get(rotulo, 730)
    if dias is None:
        return "2000-01-01"
    return (dt.date.today() - dt.timedelta(days=dias)).isoformat()
