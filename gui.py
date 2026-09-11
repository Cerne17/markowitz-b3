"""Dashboard customtkinter: alocacao otima de carteira B3 via Markowitz + Sharpe."""
import json
import queue
import threading
from tkinter import messagebox, ttk

import customtkinter as ctk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import data_fetcher as dfx
import portfolio_optimizer as opt

# ---------- paleta de marca (cerne.pro - "heartwood core") ----------
INK = "#0C0D10"
SURFACE = "#14161B"
SURFACE_2 = "#1F2229"
TEXT = "#E8E4DB"
TEXT_MUTED = "#9A968C"
HEARTWOOD = "#E89A3C"
HEARTWOOD_GLOW = "#F6B65A"
OXBLOOD = "#8C3B24"
SAPWOOD = "#4E8F6B"
POSITIVO = "#6FBF97"  # sapwood clareado p/ contraste em texto pequeno
NEGATIVO = "#E0665A"  # oxblood clareado p/ contraste em texto pequeno

# 8 matizes categoricos fixos (identidade de ativo), validados CVD/contraste
# contra a superficie SURFACE - ordem nunca muda, so aumenta em ciclo apos 8.
PALETA_CATEGORICA = [
    "#C9832A", "#2F9A6B", "#4A90D9", "#C0574A",
    "#1FA79E", "#9B6FD1", "#D1487E", "#A68A1F",
]

CMAP_SHARPE = LinearSegmentedColormap.from_list("sharpe", [SURFACE_2, HEARTWOOD, HEARTWOOD_GLOW])
CMAP_CORRELACAO = LinearSegmentedColormap.from_list("correlacao", [OXBLOOD, "#5A5852", SAPWOOD])

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

CONFIG_PATH = "portfolio.json"


def carrega_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def salva_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def cor_ativo(indice: int) -> str:
    return PALETA_CATEGORICA[indice % len(PALETA_CATEGORICA)]


MAPA_CLASSE = {"Acoes": "Acao", "ETFs": "ETF", "FIIs": "FII"}

FILTROS_REBALANCEAMENTO = {
    "Todos": None,
    "Com posicao atual (> 0)": lambda df: df[df["valor_atual"] > 0],
    "Com alocacao alvo (> 0)": lambda df: df[df["peso_alvo"] > 1e-6],
    "So compras": lambda df: df[df["cotas_sugeridas"] > 0],
    "So vendas": lambda df: df[df["cotas_sugeridas"] < 0],
}

# coluna da tabela -> coluna do DataFrame usada de fato p/ ordenar (numerica, nao o texto formatado)
MAPA_ORDENACAO_REBALANCEAMENTO = {
    "ticker": "ticker",
    "peso_atual": "peso_atual",
    "peso_alvo": "peso_alvo",
    "valor_atual": "valor_atual",
    "preco_atual": "preco_atual",
    "ajuste": "valor_transacao",
    "ir_estimado": "ir_estimado",
}

# nome exibido no dropdown -> chave do portfolio correspondente no payload calculado
ESTRATEGIAS_REFERENCIA = {
    "Max Sharpe": "carteira_max_sharpe",
    "Min Volatilidade": "carteira_min_vol",
    "Equal Weight (1/N)": "carteira_equal_weight",
    "Risk Parity": "carteira_risk_parity",
    "Max Sortino": "carteira_max_sortino",
    "Max Calmar": "carteira_max_calmar",
    "Max STARR": "carteira_max_starr",
    "Min CVaR": "carteira_min_cvar",
}

# estilo do marcador de cada estrategia no grafico da fronteira - todas aparecem juntas la,
# mesmo so uma sendo escolhida como "Referencia" no Resumo por vez
GLOSSARIO = [
    ("Fronteira Eficiente (Teoria de Markowitz)",
     "Pra cada nivel de risco (volatilidade), a fronteira mostra a maior combinacao de ativos "
     "que entrega o maior retorno esperado possivel - carteiras abaixo dela sao ineficientes "
     "(dava pra ganhar mais com o mesmo risco). Cada ponto da curva branca no grafico e uma "
     "carteira otima diferente; a nuvem colorida ao redor sao milhares de combinacoes "
     "aleatorias de peso, coloridas pelo indice de Sharpe. Clique em qualquer ponto (curva ou "
     "marcador) pra usar aquela alocacao como alvo de rebalanceamento."),
    ("Indice de Sharpe",
     "Retorno acima da taxa livre de risco, dividido pela volatilidade total: "
     "(retorno - taxa livre) / volatilidade. Quanto maior, melhor - mas penaliza toda a "
     "volatilidade, inclusive a boa (dias em que o ativo sobe muito tambem contam como risco)."),
    ("Indice de Sortino",
     "Parecido com o Sharpe, mas so penaliza a volatilidade ruim (retornos abaixo da taxa "
     "livre de risco) - ignora a variacao positiva. Faz mais sentido pra quem se importa so "
     "com perdas, nao com toda oscilacao do preco."),
    ("Indice de Calmar",
     "Retorno anual dividido pelo maior drawdown historico (a pior queda de pico a vale que a "
     "carteira sofreu). Muito citado em relatorios de fundos porque fala a lingua do "
     "investidor: quanto voce ganha por unidade da pior dor que ja sentiu."),
    ("STARR Ratio",
     "Como o Sharpe, mas troca a volatilidade pelo CVaR (a perda media nos piores cenarios "
     "historicos) - foco em risco de cauda (eventos raros e ruins) em vez de dispersao geral "
     "simetrica."),
    ("Value at Risk (VaR) e CVaR",
     "VaR (95%) e a perda que a carteira nao deve ultrapassar em 95% dos dias, com base no "
     "historico. CVaR e a perda MEDIA justamente nos 5% de dias piores que esse limite - mais "
     "conservador, porque nao ignora o tamanho da cauda, so a frequencia dela."),
    ("Max Drawdown",
     "A maior queda percentual entre um pico e o vale seguinte no historico da carteira. "
     "Mostra o pior momento psicologico que quem segurou a carteira teria vivido - relevante "
     "pra saber se voce aguentaria ver seu patrimonio cair aquilo."),
    ("Equal Weight (1/N)",
     "Mesmo peso pra cada ativo, sem otimizacao nenhuma. Parece ingenuo, mas estudos classicos "
     "(DeMiguel, Garlappi & Uppal, 2009) mostram que costuma superar carteiras 'otimizadas' "
     "fora da amostra, porque nao depende de estimativas de retorno esperado - que sao muito "
     "ruidosas na pratica."),
    ("Risk Parity",
     "Em vez de pesos iguais em R$, iguala a CONTRIBUICAO DE RISCO de cada ativo pra variancia "
     "total da carteira - um ativo mais volatil recebe menos peso, um mais estavel recebe "
     "mais. Nao usa retorno esperado no calculo, so a matriz de covariancia, entao e menos "
     "sensivel a erro de estimativa de retorno."),
    ("Rebalanceamento e Imposto de Renda",
     "A sugestao de rebalanceamento mostra quantas cotas comprar/vender pra chegar no alvo "
     "escolhido, considerando sua carteira atual + o aporte disponivel. A coluna de IR "
     "estimado indica a aliquota que se aplicaria numa venda: acoes tem isencao se o total "
     "vendido no mes for ate R$20 mil; ETFs e FIIs nao tem essa isencao (15% e 20% "
     "respectivamente). E so uma estimativa - o app nao sabe seu preco medio de compra, "
     "entao nao calcula o ganho de capital real."),
    ("Renda Passiva (Dividendos)",
     "Projecao de quanto voce receberia por mes em dividendos, com base no dividend yield "
     "atual de cada ativo (dado do Yahoo Finance) aplicado ao valor do alvo escolhido. "
     "Dividendos de acoes e FIIs sao isentos de IR p/ pessoa fisica; Juros sobre Capital "
     "Proprio (JCP) tem retencao de 15% na fonte."),
    ("Buy and Hold",
     "A filosofia do app: voce define a alocacao alvo e a mantem, sem ficar comprando e "
     "vendendo por timing de mercado. O rebalanceamento existe pra dizer quanto ajustar quando "
     "voce tem dinheiro novo pra investir (aporte) - nao pra sugerir trades frequentes."),
]

ESTILOS_MARCADOR_ESTRATEGIA = {
    "Max Sharpe": {"color": "#E89A3C", "marker": "*", "s": 240},
    "Min Volatilidade": {"color": "#4E8F6B", "marker": "*", "s": 200},
    "Equal Weight (1/N)": {"color": "#9B6FD1", "marker": "s", "s": 110},
    "Risk Parity": {"color": "#1FA79E", "marker": "^", "s": 130},
    "Max Sortino": {"color": "#D1487E", "marker": "P", "s": 140},
    "Max Calmar": {"color": "#A68A1F", "marker": "X", "s": 140},
    "Max STARR": {"color": "#C0574A", "marker": "p", "s": 140},
    "Min CVaR": {"color": "#2F9A6B", "marker": "h", "s": 130},
}


# larguras fixas compartilhadas entre o cabecalho e cada LinhaAtivo, p/ colunas alinhadas
LARGURA_COL_TICKER = 78
LARGURA_COL_VALOR = 104
LARGURA_COL_REMOVER = 72


class LinhaAtivo(ctk.CTkFrame):
    def __init__(self, master, ticker: str, valor: float, on_remover):
        super().__init__(master, fg_color="transparent")
        self.entry_ticker = ctk.CTkEntry(self, placeholder_text="PETR4", width=LARGURA_COL_TICKER,
                                          fg_color=SURFACE, border_color=TEXT_MUTED, border_width=1,
                                          text_color=TEXT, placeholder_text_color=TEXT_MUTED)
        self.entry_ticker.insert(0, ticker)
        self.entry_ticker.grid(row=0, column=0, padx=(0, 4), pady=3)

        self.entry_valor = ctk.CTkEntry(self, placeholder_text="0,00", width=LARGURA_COL_VALOR,
                                         fg_color=SURFACE, border_color=TEXT_MUTED, border_width=1,
                                         text_color=TEXT, placeholder_text_color=TEXT_MUTED)
        self.entry_valor.insert(0, f"{valor:.2f}")
        self.entry_valor.grid(row=0, column=1, padx=4, pady=3)

        btn_remover = ctk.CTkButton(self, text="Remover", width=LARGURA_COL_REMOVER,
                                     fg_color=OXBLOOD, hover_color="#A8482D",
                                     text_color=TEXT, command=lambda: on_remover(self))
        btn_remover.grid(row=0, column=2, padx=(4, 0), pady=3)

    def get_dados(self):
        ticker = self.entry_ticker.get().strip()
        valor_txt = self.entry_valor.get().strip().replace(",", ".")
        if not ticker or not valor_txt:
            return None
        return ticker, float(valor_txt)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Carteira B3 - Markowitz & Sharpe")
        self.geometry("1320x820")
        self.minsize(1080, 680)
        self.configure(fg_color=INK)

        self.fila = queue.Queue()
        self.linhas_ativos: list[LinhaAtivo] = []
        self.resultado = None
        self.pontos_fronteira: list[opt.Portfolio] = []
        self.pontos_candidatos: list[opt.Portfolio] = []
        self.df_rebalanceamento: pd.DataFrame | None = None
        self._ordenacao_rebalanceamento: tuple[str, bool] = ("ticker", False)
        self.escolhida: opt.Portfolio | None = None
        self.universo_ativos = dfx.carrega_universo_ativos()
        self.ativo_selecionado_explorar: dict | None = None
        self._info_ativo_atual: dict = {}

        self._monta_layout()
        self._carrega_config_na_ui()
        self.after(150, self._processa_fila)

    # ---------- layout ----------
    def _monta_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._monta_painel_esquerdo()
        self._monta_area_direita()

    def _monta_painel_esquerdo(self):
        # CTkScrollableFrame p/ o painel inteiro rolar quando a janela for baixa demais
        # p/ caber tudo (ex: botao "Calcular" ficando escondido sem como rolar ate ele).
        painel = ctk.CTkScrollableFrame(self, width=340, fg_color=SURFACE,
                                         scrollbar_button_color=SURFACE_2,
                                         scrollbar_button_hover_color=HEARTWOOD)
        painel.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)

        ctk.CTkLabel(painel, text="Minha Carteira", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=14, pady=(14, 2))
        ctk.CTkLabel(painel, text="Valor em R$ investido em cada ativo (nao e qtd. de acoes/cotas). "
                                   "Use 0 p/ incluir na analise um ativo que ainda nao tem.",
                     font=ctk.CTkFont(size=11), text_color=TEXT_MUTED, wraplength=310, justify="left").pack(
            anchor="w", padx=14, pady=(0, 8))

        cabecalho = ctk.CTkFrame(painel, fg_color="transparent")
        cabecalho.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkLabel(cabecalho, text="Ticker", width=LARGURA_COL_TICKER, anchor="w",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).grid(
            row=0, column=0, padx=(0, 4))
        ctk.CTkLabel(cabecalho, text="Valor (R$)", width=LARGURA_COL_VALOR, anchor="w",
                     font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).grid(
            row=0, column=1, padx=4)
        ctk.CTkLabel(cabecalho, text="", width=LARGURA_COL_REMOVER).grid(row=0, column=2, padx=(4, 0))

        self.frame_ativos = ctk.CTkFrame(painel, fg_color=SURFACE_2)
        self.frame_ativos.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkButton(painel, text="+ Adicionar ativo", fg_color=SURFACE_2, hover_color="#2A2E38",
                      text_color=TEXT, command=self._adiciona_linha_vazia).pack(
            fill="x", padx=14, pady=(4, 12))

        ctk.CTkLabel(painel, text="Aporte disponivel agora (R$) - opcional", text_color=TEXT_MUTED).pack(
            anchor="w", padx=14)
        self.entry_aporte = ctk.CTkEntry(painel, fg_color=SURFACE_2, border_color=TEXT_MUTED, border_width=1,
                                          text_color=TEXT, placeholder_text="0,00")
        self.entry_aporte.pack(fill="x", padx=14, pady=(0, 10))
        self.entry_aporte.bind("<KeyRelease>", lambda e: self._recalcula_tabela_rebalanceamento())

        ctk.CTkLabel(painel, text="Data inicio (AAAA-MM-DD)", text_color=TEXT_MUTED).pack(anchor="w", padx=14)
        self.entry_data_inicio = ctk.CTkEntry(painel, fg_color=SURFACE_2, border_color=TEXT_MUTED, border_width=1,
                                               text_color=TEXT)
        self.entry_data_inicio.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(painel, text="Taxa livre de risco anual (%)", text_color=TEXT_MUTED).pack(anchor="w", padx=14)
        self.entry_taxa_livre = ctk.CTkEntry(painel, fg_color=SURFACE_2, border_color=TEXT_MUTED, border_width=1,
                                              text_color=TEXT)
        self.entry_taxa_livre.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(painel, text="Simulacoes de portfolio (fronteira)", text_color=TEXT_MUTED).pack(
            anchor="w", padx=14)
        self.entry_num_sim = ctk.CTkEntry(painel, fg_color=SURFACE_2, border_color=TEXT_MUTED, border_width=1,
                                           text_color=TEXT)
        self.entry_num_sim.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(painel, text="Peso maximo por ativo (%) - evita concentracao total",
                     text_color=TEXT_MUTED, wraplength=300, justify="left").pack(anchor="w", padx=14)
        self.entry_peso_maximo = ctk.CTkEntry(painel, fg_color=SURFACE_2, border_color=TEXT_MUTED, border_width=1,
                                               text_color=TEXT)
        self.entry_peso_maximo.pack(fill="x", padx=14, pady=(0, 14))

        self.btn_calcular = ctk.CTkButton(painel, text="Calcular alocacao otima", height=42,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color=HEARTWOOD, hover_color=HEARTWOOD_GLOW, text_color=INK,
                                           command=self._on_calcular)
        self.btn_calcular.pack(fill="x", padx=14, pady=(0, 6))

        self.label_status = ctk.CTkLabel(painel, text="", text_color=TEXT_MUTED, wraplength=300, justify="left")
        self.label_status.pack(fill="x", padx=14, pady=(4, 14))

    def _monta_area_direita(self):
        self.tabview = ctk.CTkTabview(self, fg_color=SURFACE, segmented_button_fg_color=SURFACE_2,
                                       segmented_button_selected_color=HEARTWOOD,
                                       segmented_button_selected_hover_color=HEARTWOOD_GLOW,
                                       segmented_button_unselected_color=SURFACE_2,
                                       text_color=TEXT, text_color_disabled=TEXT_MUTED)
        self.tabview.grid(row=0, column=1, sticky="nswe", padx=(6, 12), pady=12)

        self.tab_explorar = self.tabview.add("Explorar Ativos")
        self.tab_resumo = self.tabview.add("Resumo")
        self.tab_fronteira = self.tabview.add("Fronteira Eficiente")
        self.tab_alocacao = self.tabview.add("Alocacao")
        self.tab_risco = self.tabview.add("Risco & Desempenho")
        self.tab_correlacao = self.tabview.add("Correlacao & Retornos")
        self.tab_guia = self.tabview.add("Guia")

        self._monta_tab_explorar()
        self._monta_tab_resumo()
        self._monta_canvas(self.tab_fronteira, "fronteira")
        self._monta_canvas(self.tab_alocacao, "alocacao")
        self._monta_tab_risco()
        self._monta_canvas(self.tab_correlacao, "correlacao")
        self._monta_tab_guia()

        self.canvas_fronteira.mpl_connect("button_press_event", self._on_click_fronteira)

    def _monta_tab_explorar(self):
        tab = self.tab_explorar
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        painel = ctk.CTkFrame(tab, width=320, fg_color=SURFACE_2)
        painel.grid(row=0, column=0, sticky="nswe", padx=(8, 4), pady=8)
        painel.grid_propagate(False)

        ctk.CTkLabel(painel, text="Buscar Ativos", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=12, pady=(12, 6))

        ctk.CTkLabel(painel, text="Ticker especifico (nao esta na lista abaixo?)",
                     font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(anchor="w", padx=12)
        frame_manual = ctk.CTkFrame(painel, fg_color="transparent")
        frame_manual.pack(fill="x", padx=12, pady=(0, 10))
        self.entry_ticker_manual = ctk.CTkEntry(frame_manual, placeholder_text="Ex: AURA33", fg_color=SURFACE,
                                                 border_color=TEXT_MUTED, border_width=1, text_color=TEXT,
                                                 placeholder_text_color=TEXT_MUTED)
        self.entry_ticker_manual.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry_ticker_manual.bind("<Return>", lambda e: self._buscar_ticker_manual())
        ctk.CTkButton(frame_manual, text="Buscar", width=64, fg_color=HEARTWOOD, hover_color=HEARTWOOD_GLOW,
                      text_color=INK, command=self._buscar_ticker_manual).pack(side="left")

        ctk.CTkLabel(painel, text="...ou filtre a amostra curada por setor/classe:",
                     font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(0, 4))

        classes = ["Todas"] + [c for c in ("Acoes", "ETFs", "FIIs")
                                if MAPA_CLASSE[c] in {a["classe"] for a in self.universo_ativos}]
        self.opcao_classe = ctk.CTkOptionMenu(painel, values=classes, fg_color=SURFACE,
                                               button_color=HEARTWOOD, button_hover_color=HEARTWOOD_GLOW,
                                               text_color=TEXT, dropdown_fg_color=SURFACE,
                                               dropdown_text_color=TEXT,
                                               command=lambda _: self._on_muda_classe())
        self.opcao_classe.set("Todas")
        self.opcao_classe.pack(fill="x", padx=12, pady=(0, 8))

        self.opcao_setor = ctk.CTkOptionMenu(painel, values=["Todos"], fg_color=SURFACE,
                                              button_color=HEARTWOOD, button_hover_color=HEARTWOOD_GLOW,
                                              text_color=TEXT, dropdown_fg_color=SURFACE,
                                              dropdown_text_color=TEXT,
                                              command=lambda _: self._filtra_lista_ativos())
        self.opcao_setor.set("Todos")
        self.opcao_setor.pack(fill="x", padx=12, pady=(0, 8))

        self.entry_busca = ctk.CTkEntry(painel, placeholder_text="Nome ou ticker...", fg_color=SURFACE,
                                         border_color=TEXT_MUTED, border_width=1,
                                         text_color=TEXT, placeholder_text_color=TEXT_MUTED)
        self.entry_busca.pack(fill="x", padx=12, pady=(0, 8))
        self.entry_busca.bind("<KeyRelease>", lambda e: self._filtra_lista_ativos())

        ctk.CTkLabel(painel, text="Horizonte (grafico e metricas)", font=ctk.CTkFont(size=11),
                     text_color=TEXT_MUTED).pack(anchor="w", padx=12)
        self.opcao_horizonte = ctk.CTkOptionMenu(painel, values=list(dfx.HORIZONTES_EXPLORAR.keys()),
                                                  fg_color=SURFACE, button_color=HEARTWOOD,
                                                  button_hover_color=HEARTWOOD_GLOW, text_color=TEXT,
                                                  dropdown_fg_color=SURFACE, dropdown_text_color=TEXT,
                                                  command=lambda _: self._on_muda_horizonte())
        self.opcao_horizonte.set("2 anos")
        self.opcao_horizonte.pack(fill="x", padx=12, pady=(0, 8))

        self.frame_lista_ativos = ctk.CTkScrollableFrame(painel, fg_color=SURFACE)
        self.frame_lista_ativos.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        detalhe = ctk.CTkFrame(tab, fg_color=SURFACE_2)
        detalhe.grid(row=0, column=1, sticky="nswe", padx=(4, 8), pady=8)
        detalhe.grid_rowconfigure(3, weight=1)
        detalhe.grid_columnconfigure(0, weight=1)

        self.label_titulo_ativo = ctk.CTkLabel(detalhe, text="Selecione um ativo na lista a esquerda",
                                                font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT)
        self.label_titulo_ativo.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))

        self.label_info_ativo = ctk.CTkLabel(detalhe, text="", justify="left", text_color=TEXT_MUTED,
                                              wraplength=700)
        self.label_info_ativo.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 4))

        self.label_stats_ativo = ctk.CTkLabel(detalhe, text="", justify="left",
                                               font=ctk.CTkFont(size=13), text_color=TEXT)
        self.label_stats_ativo.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 8))

        fig = Figure(figsize=(8, 3.6), dpi=100, facecolor=INK)
        canvas = FigureCanvasTkAgg(fig, master=detalhe)
        canvas.get_tk_widget().grid(row=3, column=0, sticky="nswe", padx=16, pady=(0, 8))
        self.fig_explorar = fig
        self.canvas_explorar = canvas

        linha_add = ctk.CTkFrame(detalhe, fg_color="transparent")
        linha_add.grid(row=4, column=0, sticky="we", padx=16, pady=(0, 16))
        ctk.CTkLabel(linha_add, text="Valor a investir (R$)", text_color=TEXT_MUTED).pack(side="left", padx=(0, 8))
        self.entry_valor_novo_ativo = ctk.CTkEntry(linha_add, width=140, fg_color=SURFACE,
                                                     border_color=TEXT_MUTED, border_width=1, text_color=TEXT)
        self.entry_valor_novo_ativo.insert(0, "1000.00")
        self.entry_valor_novo_ativo.pack(side="left", padx=(0, 8))
        self.btn_adicionar_ativo = ctk.CTkButton(linha_add, text="+ Adicionar ao Portfolio",
                                                  fg_color=HEARTWOOD, hover_color=HEARTWOOD_GLOW, text_color=INK,
                                                  font=ctk.CTkFont(weight="bold"), state="disabled",
                                                  command=self._adicionar_ativo_ao_portfolio)
        self.btn_adicionar_ativo.pack(side="left")

        self.frame_salvar_curada = ctk.CTkFrame(detalhe, fg_color="transparent")
        self.frame_salvar_curada.grid(row=5, column=0, sticky="we", padx=16, pady=(0, 16))
        ctk.CTkLabel(self.frame_salvar_curada, text="Nao esta na lista curada -", text_color=TEXT_MUTED).pack(
            side="left", padx=(0, 8))
        self.opcao_classe_salvar = ctk.CTkOptionMenu(self.frame_salvar_curada, values=["Acoes", "ETFs", "FIIs"],
                                                      width=90, fg_color=SURFACE, button_color=SURFACE_2,
                                                      button_hover_color=HEARTWOOD, text_color=TEXT,
                                                      dropdown_fg_color=SURFACE, dropdown_text_color=TEXT)
        self.opcao_classe_salvar.pack(side="left", padx=(0, 6))
        self.entry_setor_salvar = ctk.CTkEntry(self.frame_salvar_curada, placeholder_text="Setor", width=170,
                                                fg_color=SURFACE, border_color=TEXT_MUTED, border_width=1,
                                                text_color=TEXT, placeholder_text_color=TEXT_MUTED)
        self.entry_setor_salvar.pack(side="left", padx=(0, 6))
        ctk.CTkButton(self.frame_salvar_curada, text="+ Salvar na lista", fg_color=SAPWOOD, hover_color=POSITIVO,
                      text_color=INK, command=self._salvar_ativo_na_lista).pack(side="left")
        self.frame_salvar_curada.grid_remove()

        self._filtra_lista_ativos()

    def _on_muda_classe(self):
        classe_label = self.opcao_classe.get()
        classe = MAPA_CLASSE.get(classe_label)
        if classe is None:
            setores = sorted({a["setor"] for a in self.universo_ativos})
        else:
            setores = sorted({a["setor"] for a in self.universo_ativos if a["classe"] == classe})
        self.opcao_setor.configure(values=["Todos"] + setores)
        self.opcao_setor.set("Todos")
        self._filtra_lista_ativos()

    def _filtra_lista_ativos(self):
        classe = MAPA_CLASSE.get(self.opcao_classe.get())
        setor = self.opcao_setor.get()
        termo = self.entry_busca.get().strip().lower()
        resultado = []
        for a in self.universo_ativos:
            if classe is not None and a["classe"] != classe:
                continue
            if setor != "Todos" and a["setor"] != setor:
                continue
            if termo and termo not in a["nome"].lower() and termo not in a["ticker"].lower():
                continue
            resultado.append(a)
        self._popula_lista_ativos(resultado)

    def _popula_lista_ativos(self, lista):
        for widget in self.frame_lista_ativos.winfo_children():
            widget.destroy()
        if not lista:
            ctk.CTkLabel(self.frame_lista_ativos, text="Nenhum ativo encontrado.",
                         text_color=TEXT_MUTED).pack(pady=8)
            return
        for a in lista:
            prefixo = "" if a["classe"] == "Acao" else f"[{a['classe']}] "
            texto = f"{prefixo}{a['ticker'].replace('.SA', '')} · {a['nome']}"
            btn = ctk.CTkButton(self.frame_lista_ativos, text=texto, anchor="w",
                                 fg_color=SURFACE_2, hover_color=HEARTWOOD, text_color=TEXT,
                                 command=lambda a=a: self._selecionar_ativo_explorar(a))
            btn.pack(fill="x", pady=2)

    def _buscar_ticker_manual(self):
        bruto = self.entry_ticker_manual.get().strip()
        if not bruto:
            return
        ticker_norm = dfx.normaliza_ticker(bruto)

        conhecido = next((a for a in self.universo_ativos if a["ticker"] == ticker_norm), None)
        ativo = conhecido or {
            "ticker": ticker_norm,
            "nome": ticker_norm.replace(".SA", ""),
            "setor": "Nao listado na amostra curada",
            "classe": "Outro",
        }
        self._selecionar_ativo_explorar(ativo)

    def _salvar_ativo_na_lista(self):
        ativo = self.ativo_selecionado_explorar
        if ativo is None:
            return

        classe = MAPA_CLASSE.get(self.opcao_classe_salvar.get(), "Acao")
        setor = self.entry_setor_salvar.get().strip() or "Outro"
        nome = self._info_ativo_atual.get("nome_longo") or ativo["nome"]

        novo = {"ticker": ativo["ticker"], "nome": nome, "setor": setor, "classe": classe}
        dfx.salvar_ativo_no_universo(novo)

        self.universo_ativos = dfx.carrega_universo_ativos()
        self.ativo_selecionado_explorar = novo
        self._on_muda_classe()
        self.frame_salvar_curada.grid_remove()
        self.label_info_ativo.configure(
            text=self.label_info_ativo.cget("text") + f"\n\n{novo['ticker']} salvo na lista curada ({setor}).")

    def _on_muda_horizonte(self):
        if self.ativo_selecionado_explorar is not None:
            self._selecionar_ativo_explorar(self.ativo_selecionado_explorar)

    def _selecionar_ativo_explorar(self, ativo: dict):
        self.ativo_selecionado_explorar = ativo
        self.label_titulo_ativo.configure(text=f"{ativo['nome']} ({ativo['ticker']}) - {ativo['setor']}")
        self.label_info_ativo.configure(text="Carregando dados do Yahoo Finance...")
        self.label_stats_ativo.configure(text="")
        self.btn_adicionar_ativo.configure(state="disabled")
        self.frame_salvar_curada.grid_remove()
        self.fig_explorar.clear()
        self.canvas_explorar.draw()

        thread = threading.Thread(target=self._worker_analise_ativo, args=(ativo,), daemon=True)
        thread.start()

    def _worker_analise_ativo(self, ativo: dict):
        try:
            data_inicio = dfx.data_inicio_por_horizonte(self.opcao_horizonte.get())
            taxa_livre = float(self.entry_taxa_livre.get().strip().replace(",", ".")) / 100
            precos = dfx.baixar_precos([ativo["ticker"]], data_inicio)
            serie = precos[ativo["ticker"]]
            stats = opt.estatisticas_ativo_individual(serie, taxa_livre)
            info = dfx.obter_info_ativo(ativo["ticker"])
            self.fila.put(("ativo_ok", {"ativo": ativo, "serie": serie, "stats": stats, "info": info}))
        except Exception as e:
            self.fila.put(("ativo_erro", str(e)))

    def _atualiza_analise_ativo(self, payload: dict):
        info = payload["info"]
        stats = payload["stats"]
        ativo = payload["ativo"]

        self._info_ativo_atual = info

        if ativo["classe"] == "Outro":
            if info.get("nome_longo"):
                self.label_titulo_ativo.configure(text=f"{info['nome_longo']} ({ativo['ticker']})")
            if info.get("price_to_book") is None and info.get("market_cap") is None:
                self.opcao_classe_salvar.set("ETFs")
            elif info.get("setor") == "Real Estate":
                self.opcao_classe_salvar.set("FIIs")
            else:
                self.opcao_classe_salvar.set("Acoes")
            self.entry_setor_salvar.delete(0, "end")
            self.entry_setor_salvar.insert(0, info.get("setor") or "")
            self.frame_salvar_curada.grid()

        linhas_info = []
        if info.get("setor"):
            industria = f" / {info['industria']}" if info.get("industria") else ""
            linhas_info.append(f"Setor (Yahoo): {info['setor']}{industria}")
        if info.get("market_cap"):
            linhas_info.append(f"TAM (valor de mercado): R$ {info['market_cap'] / 1e9:.2f} bi")
        if info.get("price_to_book") is not None:
            linhas_info.append(f"P/VP: {info['price_to_book']:.2f}")
        if info.get("dividend_yield") is not None:
            linhas_info.append(f"Dividend yield: {info['dividend_yield']:.2f}%")
        if info.get("capex_recente") is not None:
            linhas_info.append(f"Capex mais recente ({info.get('capex_data', 'n/d')}): "
                                f"R$ {abs(info['capex_recente']) / 1e6:.1f} mi")
        if info.get("minima_52_sem") and info.get("maxima_52_sem"):
            linhas_info.append(f"Faixa 52 semanas: R$ {info['minima_52_sem']:.2f} - R$ {info['maxima_52_sem']:.2f}")
        if not info.get("market_cap") and not info.get("price_to_book"):
            linhas_info.append("TAM/P-VP nao disponiveis p/ esta classe de ativo (comum p/ ETFs).")
        self.label_info_ativo.configure(text="\n".join(linhas_info) or "Sem dados fundamentalistas disponiveis.")

        texto_stats = (
            f"Retorno anualizado: {stats['retorno_anual'] * 100:.2f}%\n"
            f"Volatilidade anualizada: {stats['volatilidade_anual'] * 100:.2f}%\n"
            f"Indice de Sharpe (individual): {stats['sharpe']:.3f}\n"
            f"Max drawdown no periodo: {stats['max_drawdown'] * 100:.2f}%"
        )
        self.label_stats_ativo.configure(text=texto_stats)

        self._desenha_grafico_explorar(payload["ativo"]["ticker"], payload["serie"])
        self.btn_adicionar_ativo.configure(state="normal")

    def _desenha_grafico_explorar(self, ticker: str, serie):
        fig = self.fig_explorar
        fig.clear()
        ax = fig.add_subplot(111)
        self._estiliza_eixos(ax)
        ax.plot(serie.index, serie.values, color=HEARTWOOD, linewidth=1.4)
        ax.set_title(f"Preco - {ticker} (Buy and Hold) - {self.opcao_horizonte.get()}", color=TEXT)
        ax.set_ylabel("R$")
        fig.tight_layout()
        self.canvas_explorar.draw()

    def _adicionar_ativo_ao_portfolio(self):
        ativo = self.ativo_selecionado_explorar
        if ativo is None:
            return
        try:
            valor = float(self.entry_valor_novo_ativo.get().strip().replace(",", "."))
        except ValueError:
            messagebox.showerror("Valor invalido", "Informe um valor numerico para investir.")
            return
        if valor < 0:
            messagebox.showerror("Valor invalido", "O valor nao pode ser negativo.")
            return

        ticker_norm = dfx.normaliza_ticker(ativo["ticker"])
        existentes = []
        for linha in self.linhas_ativos:
            dados = linha.get_dados()
            if dados:
                existentes.append(dfx.normaliza_ticker(dados[0]))
        if ticker_norm in existentes:
            messagebox.showinfo("Ja na carteira", f"{ticker_norm} ja esta na sua carteira.")
            return

        self._adiciona_linha(ticker_norm, valor)
        self.label_status.configure(text=f"{ticker_norm} adicionado. Va em Resumo e clique em Calcular.",
                                     text_color=POSITIVO)
        self.tabview.set("Resumo")

    def _monta_tab_resumo(self):
        self.tab_resumo.grid_columnconfigure((0, 1, 2), weight=1)

        linha_comparar = ctk.CTkFrame(self.tab_resumo, fg_color="transparent")
        linha_comparar.grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 0))
        ctk.CTkLabel(linha_comparar, text="Comparar carteira atual com:", text_color=TEXT_MUTED).pack(
            side="left", padx=(0, 8))
        self.opcao_estrategia_referencia = ctk.CTkOptionMenu(
            linha_comparar, values=list(ESTRATEGIAS_REFERENCIA.keys()), width=200,
            fg_color=SURFACE_2, button_color=HEARTWOOD, button_hover_color=HEARTWOOD_GLOW,
            text_color=TEXT, dropdown_fg_color=SURFACE, dropdown_text_color=TEXT,
            command=lambda _: self._atualiza_card_referencia())
        self.opcao_estrategia_referencia.set("Max Sharpe")
        self.opcao_estrategia_referencia.pack(side="left")

        self.card_atual = self._cria_card(self.tab_resumo, "Carteira Atual", 0, linha=1)
        self.card_referencia = self._cria_card(self.tab_resumo, "Referencia: Max Sharpe", 1, linha=1)
        self.card_escolhida = self._cria_card(self.tab_resumo, "Alvo Escolhido na Fronteira", 2, linha=1,
                                               destaque=True)

        linha_titulo = ctk.CTkFrame(self.tab_resumo, fg_color="transparent")
        linha_titulo.grid(row=2, column=0, columnspan=3, sticky="we", padx=16, pady=(18, 4))
        linha_titulo.grid_columnconfigure(0, weight=1)

        self.label_titulo_rebalanceamento = ctk.CTkLabel(
            linha_titulo,
            text="Sugestao de rebalanceamento (clique num ponto da Fronteira Eficiente p/ mudar o alvo)",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT, wraplength=700, justify="left")
        self.label_titulo_rebalanceamento.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(linha_titulo, text="Filtro:", text_color=TEXT_MUTED).grid(row=0, column=1, padx=(8, 4))
        self.opcao_filtro_rebalanceamento = ctk.CTkOptionMenu(
            linha_titulo, values=list(FILTROS_REBALANCEAMENTO.keys()), width=200,
            fg_color=SURFACE_2, button_color=HEARTWOOD, button_hover_color=HEARTWOOD_GLOW,
            text_color=TEXT, dropdown_fg_color=SURFACE, dropdown_text_color=TEXT,
            command=lambda _: self._renderiza_tabela_rebalanceamento())
        self.opcao_filtro_rebalanceamento.set("Todos")
        self.opcao_filtro_rebalanceamento.grid(row=0, column=2)

        estilo_tv = ttk.Style()
        estilo_tv.theme_use("default")
        estilo_tv.configure("Treeview", background=SURFACE_2, fieldbackground=SURFACE_2,
                             foreground=TEXT, rowheight=26, borderwidth=0)
        estilo_tv.configure("Treeview.Heading", background=INK, foreground=TEXT)
        estilo_tv.map("Treeview", background=[("selected", HEARTWOOD)], foreground=[("selected", INK)])

        colunas = ("ticker", "peso_atual", "peso_alvo", "valor_atual", "preco_atual", "ajuste", "ir_estimado")
        self.tabela_rebalanceamento = ttk.Treeview(self.tab_resumo, columns=colunas, show="headings", height=8)
        self._titulos_colunas_rebalanceamento = {
            "ticker": "Ticker", "peso_atual": "Peso atual", "peso_alvo": "Peso alvo",
            "valor_atual": "Valor atual (R$)", "preco_atual": "Preco/cota (R$)",
            "ajuste": "Ajuste sugerido", "ir_estimado": "IR estimado (venda)"}
        larguras = {"ajuste": 240}
        for c in colunas:
            self.tabela_rebalanceamento.heading(c, text=self._titulos_colunas_rebalanceamento[c],
                                                 command=lambda c=c: self._ordenar_tabela_rebalanceamento(c))
            self.tabela_rebalanceamento.column(c, anchor="center", width=larguras.get(c, 120))
        self.tabela_rebalanceamento.tag_configure("comprar", foreground=POSITIVO)
        self.tabela_rebalanceamento.tag_configure("vender", foreground=NEGATIVO)
        self.tabela_rebalanceamento.grid(row=3, column=0, columnspan=3, sticky="nswe", padx=16, pady=6)
        self.tab_resumo.grid_rowconfigure(3, weight=1)

        self.label_nota_ir = ctk.CTkLabel(
            self.tab_resumo,
            text=("Cotas arredondadas p/ numero inteiro (preco do ultimo fechamento) - o valor real "
                  "da transacao pode diferir um pouco do ajuste teorico. IR estimado e simplificado: nao "
                  "calcula o ganho de capital real (o app nao rastreia seu preco medio de compra), so "
                  "indica a aliquota/isencao que se aplicaria na venda. Consulte um contador antes de decidir."),
            font=ctk.CTkFont(size=10), text_color=TEXT_MUTED, wraplength=900, justify="left")
        self.label_nota_ir.grid(row=4, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 6))

        self.label_renda_passiva = ctk.CTkLabel(
            self.tab_resumo, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color=SAPWOOD,
            justify="left", wraplength=900)
        self.label_renda_passiva.grid(row=5, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 14))

    def _cria_card(self, master, titulo, coluna, linha=0, destaque=False):
        card = ctk.CTkFrame(master, fg_color=SURFACE_2,
                             border_width=2 if destaque else 0,
                             border_color=HEARTWOOD if destaque else SURFACE_2)
        card.grid(row=linha, column=coluna, sticky="nswe", padx=10, pady=14)
        titulo_label = ctk.CTkLabel(card, text=titulo, font=ctk.CTkFont(size=14, weight="bold"),
                                     text_color=HEARTWOOD if destaque else TEXT)
        titulo_label.pack(pady=(12, 6))
        label_valores = ctk.CTkLabel(card, text="--", justify="left", font=ctk.CTkFont(size=13), text_color=TEXT)
        label_valores.pack(padx=16, pady=(0, 14))
        card.label_valores = label_valores
        card.titulo_label = titulo_label
        return card

    def _monta_canvas(self, tab, chave):
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=INK)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        setattr(self, f"fig_{chave}", fig)
        setattr(self, f"canvas_{chave}", canvas)

    def _monta_tab_guia(self):
        scroll = ctk.CTkScrollableFrame(self.tab_guia, fg_color=SURFACE)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(scroll, text="Guia rapido: o que cada metrica/estrategia significa",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(
            anchor="w", padx=12, pady=(8, 4))
        ctk.CTkLabel(scroll, text="Referencia rapida dos conceitos usados no app - nao e recomendacao de "
                                   "investimento.", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(
            anchor="w", padx=12, pady=(0, 14))

        for titulo, texto in GLOSSARIO:
            ctk.CTkLabel(scroll, text=titulo, font=ctk.CTkFont(size=15, weight="bold"),
                         text_color=HEARTWOOD).pack(anchor="w", padx=12, pady=(10, 2))
            ctk.CTkLabel(scroll, text=texto, font=ctk.CTkFont(size=13), text_color=TEXT,
                         wraplength=900, justify="left").pack(anchor="w", padx=12, pady=(0, 4))
            ctk.CTkFrame(scroll, fg_color=SURFACE_2, height=1).pack(fill="x", padx=12, pady=(6, 4))

    def _monta_tab_risco(self):
        tab = self.tab_risco
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        card_risco = ctk.CTkFrame(tab, fg_color=SURFACE_2)
        card_risco.grid(row=0, column=0, sticky="we", padx=8, pady=(8, 4))
        self.label_var_cvar = ctk.CTkLabel(
            card_risco, text="Calcule a carteira p/ ver VaR/CVaR historico do alvo escolhido.",
            justify="left", text_color=TEXT, font=ctk.CTkFont(size=13), wraplength=1000)
        self.label_var_cvar.pack(anchor="w", padx=14, pady=10)

        fig = Figure(figsize=(9, 8), dpi=100, facecolor=INK)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nswe", padx=8, pady=(4, 8))
        self.fig_risco = fig
        self.canvas_risco = canvas

    # ---------- config <-> UI ----------
    def _carrega_config_na_ui(self):
        cfg = carrega_config()
        for item in cfg["ativos"]:
            self._adiciona_linha(item["ticker"], item["valor_investido"])
        self.entry_data_inicio.insert(0, cfg["data_inicio"])
        self.entry_taxa_livre.insert(0, f"{cfg['taxa_livre_risco_anual'] * 100:.2f}")
        self.entry_num_sim.insert(0, str(cfg["num_portfolios_simulados"]))
        self.entry_peso_maximo.insert(0, f"{cfg.get('peso_maximo_por_ativo', 1.0) * 100:.0f}")
        self.entry_aporte.insert(0, f"{cfg.get('aporte_disponivel', 0.0):.2f}")

    def _adiciona_linha(self, ticker="", valor=1000.0):
        linha = LinhaAtivo(self.frame_ativos, ticker, valor, self._remove_linha)
        linha.pack(fill="x", pady=2)
        self.linhas_ativos.append(linha)

    def _adiciona_linha_vazia(self):
        self._adiciona_linha("", 1000.0)

    def _remove_linha(self, linha: LinhaAtivo):
        if len(self.linhas_ativos) <= 2:
            messagebox.showwarning("Aviso", "Mantenha ao menos 2 ativos para diversificar a carteira.")
            return
        linha.destroy()
        self.linhas_ativos.remove(linha)

    def _le_formulario(self):
        ativos = []
        for linha in self.linhas_ativos:
            dados = linha.get_dados()
            if dados is None:
                continue
            ticker, valor = dados
            if valor < 0:
                raise ValueError(f"Valor investido invalido para {ticker} (nao pode ser negativo).")
            ativos.append({"ticker": dfx.normaliza_ticker(ticker), "valor_investido": valor})

        if len(ativos) < 2:
            raise ValueError("Adicione ao menos 2 ativos.")

        vistos = set()
        for a in ativos:
            if a["ticker"] in vistos:
                raise ValueError(f"Ticker repetido: {a['ticker']}")
            vistos.add(a["ticker"])

        data_inicio = self.entry_data_inicio.get().strip()
        taxa_livre = float(self.entry_taxa_livre.get().strip().replace(",", ".")) / 100
        num_sim = int(self.entry_num_sim.get().strip())
        peso_maximo = float(self.entry_peso_maximo.get().strip().replace(",", ".")) / 100
        aporte_txt = self.entry_aporte.get().strip().replace(",", ".")
        aporte = float(aporte_txt) if aporte_txt else 0.0
        if aporte < 0:
            raise ValueError("Aporte disponivel nao pode ser negativo.")
        if sum(a["valor_investido"] for a in ativos) + aporte <= 0:
            raise ValueError("Informe um valor investido em algum ativo ou um aporte maior que zero.")

        if not (1 / len(ativos) - 1e-9 <= peso_maximo <= 1.0):
            raise ValueError(
                f"Peso maximo por ativo deve estar entre {100 / len(ativos):.1f}% e 100% "
                f"para {len(ativos)} ativos.")

        return {
            "ativos": ativos,
            "data_inicio": data_inicio,
            "taxa_livre_risco_anual": taxa_livre,
            "num_portfolios_simulados": num_sim,
            "peso_maximo_por_ativo": peso_maximo,
            "aporte_disponivel": aporte,
        }

    # ---------- calculo ----------
    def _on_calcular(self):
        try:
            cfg = self._le_formulario()
        except ValueError as e:
            messagebox.showerror("Entrada invalida", str(e))
            return

        salva_config(cfg)
        self.btn_calcular.configure(state="disabled", text="Calculando...")
        self.label_status.configure(text="Baixando dados no Yahoo Finance e otimizando...", text_color=TEXT_MUTED)

        thread = threading.Thread(target=self._worker_calculo, args=(cfg,), daemon=True)
        thread.start()

    def _worker_calculo(self, cfg: dict):
        try:
            tickers = [a["ticker"] for a in cfg["ativos"]]
            valores = [a["valor_investido"] for a in cfg["ativos"]]

            tickers_com_benchmark = list(tickers)
            if dfx.BENCHMARK_IBOVESPA not in tickers_com_benchmark:
                tickers_com_benchmark.append(dfx.BENCHMARK_IBOVESPA)
            precos_completo = dfx.baixar_precos(tickers_com_benchmark, cfg["data_inicio"])

            precos = precos_completo[tickers]
            benchmark_precos = precos_completo[dfx.BENCHMARK_IBOVESPA]

            retornos = dfx.calcula_retornos_diarios(precos)
            media_anual, cov_anual = opt.estatisticas_anuais(retornos)
            taxa_livre = cfg["taxa_livre_risco_anual"]
            peso_maximo = cfg.get("peso_maximo_por_ativo", 1.0)

            if sum(valores) > 0:
                pesos_atuais = opt.pesos_atuais(valores)
            else:
                pesos_atuais = np.zeros(len(valores))  # carteira ainda vazia - so tem o aporte
            carteira_atual = opt.desempenho_portfolio(pesos_atuais, media_anual, cov_anual, taxa_livre)
            carteira_max_sharpe = opt.otimiza_max_sharpe(media_anual, cov_anual, taxa_livre, peso_maximo)
            carteira_min_vol = opt.otimiza_min_volatilidade(media_anual, cov_anual, taxa_livre, peso_maximo)
            carteira_equal_weight = opt.portfolio_equal_weight(media_anual, cov_anual, taxa_livre)
            carteira_risk_parity = opt.otimiza_risk_parity(media_anual, cov_anual, taxa_livre, peso_maximo)
            carteira_max_sortino = opt.otimiza_max_sortino(media_anual, cov_anual, retornos, taxa_livre, peso_maximo)
            carteira_max_calmar = opt.otimiza_max_calmar(media_anual, cov_anual, retornos, taxa_livre, peso_maximo)
            carteira_max_starr = opt.otimiza_max_starr(media_anual, cov_anual, retornos, taxa_livre, peso_maximo)
            carteira_min_cvar = opt.otimiza_min_cvar(media_anual, cov_anual, retornos, taxa_livre, peso_maximo)
            fronteira = opt.fronteira_eficiente(media_anual, cov_anual, taxa_livre, peso_maximo_por_ativo=peso_maximo)
            simulacoes = opt.simula_portfolios_aleatorios(
                media_anual, cov_anual, taxa_livre, n_simulacoes=cfg["num_portfolios_simulados"])
            correlacao = retornos.corr()
            precos_normalizados = precos / precos.iloc[0] * 100

            dividend_yields = dfx.obter_dividend_yields(tickers)
            classes_tickers = [self._classe_do_ticker(t) for t in tickers]
            precos_atuais = precos.iloc[-1][tickers].tolist()

            self.fila.put(("ok", {
                "tickers": tickers,
                "valores": valores,
                "classes_tickers": classes_tickers,
                "dividend_yields": dividend_yields,
                "precos_atuais": precos_atuais,
                "carteira_atual": carteira_atual,
                "carteira_max_sharpe": carteira_max_sharpe,
                "carteira_min_vol": carteira_min_vol,
                "carteira_equal_weight": carteira_equal_weight,
                "carteira_risk_parity": carteira_risk_parity,
                "carteira_max_sortino": carteira_max_sortino,
                "carteira_max_calmar": carteira_max_calmar,
                "carteira_max_starr": carteira_max_starr,
                "carteira_min_cvar": carteira_min_cvar,
                "fronteira": fronteira,
                "simulacoes": simulacoes,
                "correlacao": correlacao,
                "precos_normalizados": precos_normalizados,
                "retornos_diarios": retornos,
                "taxa_livre": taxa_livre,
                "benchmark_precos": benchmark_precos,
            }))
        except Exception as e:
            self.fila.put(("erro", str(e)))

    def _classe_do_ticker(self, ticker: str) -> str:
        encontrado = next((a for a in self.universo_ativos if a["ticker"] == ticker), None)
        return encontrado["classe"] if encontrado else "Acao"

    def _processa_fila(self):
        try:
            tipo, payload = self.fila.get_nowait()
        except queue.Empty:
            tipo = None

        if tipo is not None:
            try:
                if tipo == "erro":
                    self.btn_calcular.configure(state="normal", text="Calcular alocacao otima")
                    self.label_status.configure(text=f"Erro: {payload}", text_color=NEGATIVO)
                    messagebox.showerror("Erro ao calcular", payload)
                elif tipo == "ok":
                    self.btn_calcular.configure(state="normal", text="Calcular alocacao otima")
                    self.label_status.configure(
                        text="Calculo concluido. Clique na fronteira p/ escolher o alvo.", text_color=POSITIVO)
                    self._atualiza_ui(payload)
                elif tipo == "ativo_erro":
                    self.label_info_ativo.configure(text=f"Erro ao carregar: {payload}")
                elif tipo == "ativo_ok":
                    self._atualiza_analise_ativo(payload)
            except Exception as e:
                # Nunca deixar uma falha ao atualizar a tela travar o polling da fila
                # (senao o app fica preso em "Calculando..." pra sempre).
                self.btn_calcular.configure(state="normal", text="Calcular alocacao otima")
                self.label_status.configure(text=f"Erro ao atualizar a tela: {e}", text_color=NEGATIVO)
                messagebox.showerror("Erro inesperado ao atualizar a tela", str(e))

        self.after(150, self._processa_fila)

    # ---------- atualizacao visual ----------
    def _atualiza_ui(self, r: dict):
        self.resultado = r
        self.pontos_fronteira = r["fronteira"]
        self.escolhida = r["carteira_max_sharpe"]  # alvo padrao ate o usuario clicar na fronteira
        self.pontos_candidatos = list(r["fronteira"]) + [r[chave] for chave in ESTRATEGIAS_REFERENCIA.values()]

        self._atualiza_cards(r)
        self._atualiza_escolhida()
        self._desenha_alocacao(r)
        self._desenha_correlacao_e_retornos(r)

    def _atualiza_cards(self, r: dict):
        retornos = r["retornos_diarios"]
        taxa_livre = r["taxa_livre"]

        def texto(p, pesos_tickers=None):
            extras = opt.estatisticas_extras_portfolio(retornos, p.pesos, p.retorno_esperado, taxa_livre)
            base = (f"Retorno esperado: {p.retorno_esperado * 100:.2f}% a.a.\n"
                    f"Volatilidade: {p.volatilidade * 100:.2f}% a.a.\n"
                    f"Indice de Sharpe: {p.sharpe:.3f}\n"
                    f"Indice de Sortino: {extras['sortino']:.3f}\n"
                    f"Indice de Calmar: {extras['calmar']:.3f}\n"
                    f"Indice STARR: {extras['starr']:.3f}\n"
                    f"CVaR (95%, anualizado): {extras['cvar_anual'] * 100:.1f}%\n"
                    f"Max drawdown: {extras['max_drawdown'] * 100:.1f}%")
            if pesos_tickers:
                pesos_txt = "\n".join(f"  {t}: {w * 100:.1f}%" for t, w in pesos_tickers)
                base += f"\n\nPesos:\n{pesos_txt}"
            return base

        self.card_atual.label_valores.configure(text=texto(r["carteira_atual"]))
        self._texto_portfolio = texto  # reutilizado por _atualiza_escolhida e _atualiza_card_referencia
        self._atualiza_card_referencia()

    def _atualiza_card_referencia(self):
        if self.resultado is None:
            return
        r = self.resultado
        nome = self.opcao_estrategia_referencia.get()
        chave = ESTRATEGIAS_REFERENCIA[nome]
        portfolio = r[chave]

        self.card_referencia.titulo_label.configure(text=f"Referencia: {nome}")
        self.card_referencia.label_valores.configure(
            text=self._texto_portfolio(portfolio, list(zip(r["tickers"], portfolio.pesos))))

    # ---------- selecao interativa na fronteira ----------
    def _on_click_fronteira(self, event):
        if event.inaxes is None or not self.pontos_candidatos or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        melhor_idx, melhor_dist = 0, float("inf")
        for i, p in enumerate(self.pontos_candidatos):
            dist = (p.volatilidade * 100 - x) ** 2 + (p.retorno_esperado * 100 - y) ** 2
            if dist < melhor_dist:
                melhor_idx, melhor_dist = i, dist
        self.escolhida = self.pontos_candidatos[melhor_idx]
        self._atualiza_escolhida()

    def _ler_aporte(self) -> float:
        txt = self.entry_aporte.get().strip().replace(",", ".")
        try:
            valor = float(txt) if txt else 0.0
        except ValueError:
            valor = 0.0
        return max(valor, 0.0)

    def _atualiza_escolhida(self):
        if self.resultado is None or self.escolhida is None:
            return
        r = self.resultado
        tickers = r["tickers"]

        self.card_escolhida.label_valores.configure(
            text=self._texto_portfolio(self.escolhida, list(zip(tickers, self.escolhida.pesos))))

        self._recalcula_tabela_rebalanceamento()
        self._desenha_fronteira(r)
        self._desenha_risco(r)

    def _recalcula_tabela_rebalanceamento(self):
        if self.resultado is None or self.escolhida is None:
            return
        r = self.resultado
        aporte = self._ler_aporte()
        total = sum(r["valores"]) + aporte

        df = opt.sugestao_rebalanceamento(r["tickers"], r["valores"], self.escolhida.pesos, aporte,
                                           precos_atuais=r["precos_atuais"])
        df["classe"] = r["classes_tickers"]
        total_venda_acoes = -df.loc[(df["cotas_sugeridas"] < 0) & (df["classe"] == "Acao"), "valor_transacao"].sum()
        df["ir_estimado"] = df.apply(
            lambda row: opt.estima_aliquota_ir(row["classe"], total_venda_acoes)
            if row["cotas_sugeridas"] < 0 else "-", axis=1)

        self.df_rebalanceamento = df
        self._renderiza_tabela_rebalanceamento()
        self._atualiza_renda_passiva(r, total)

        self.label_titulo_rebalanceamento.configure(
            text=(f"Sugestao de rebalanceamento - carteira atual (R$ {sum(r['valores']):.2f}) "
                  f"+ aporte (R$ {aporte:.2f}) = total R$ {total:.2f}. "
                  "Clique num ponto da Fronteira Eficiente p/ mudar o alvo."))

    def _atualiza_renda_passiva(self, r: dict, total_carteira: float):
        tickers = r["tickers"]
        dys = r["dividend_yields"]
        pesos = self.escolhida.pesos

        renda_anual = 0.0
        sem_dado = []
        for t, w in zip(tickers, pesos):
            dy = dys.get(t)
            if dy is None:
                if w > 0:
                    sem_dado.append(t)
                continue
            renda_anual += w * total_carteira * (dy / 100)

        renda_mensal = renda_anual / 12
        texto = (f"Renda passiva estimada (dividendos, alvo escolhido): R$ {renda_mensal:.2f}/mes "
                 f"(R$ {renda_anual:.2f}/ano, ~{renda_anual / total_carteira * 100:.2f}% a.a. sobre R$ "
                 f"{total_carteira:.2f}). Dividendos sao isentos de IR p/ pessoa fisica "
                 "(JCP tem retencao de 15% na fonte).")
        if sem_dado:
            texto += f" Sem dado de yield p/: {', '.join(sem_dado)}."
        self.label_renda_passiva.configure(text=texto)

    def _ordenar_tabela_rebalanceamento(self, coluna: str):
        atual_col, atual_rev = self._ordenacao_rebalanceamento
        self._ordenacao_rebalanceamento = (coluna, not atual_rev) if atual_col == coluna else (coluna, False)
        self._renderiza_tabela_rebalanceamento()

    def _atualiza_headers_ordenacao(self):
        coluna_ativa, reverso = self._ordenacao_rebalanceamento
        indicador = " v" if reverso else " ^"
        for c, titulo in self._titulos_colunas_rebalanceamento.items():
            self.tabela_rebalanceamento.heading(c, text=titulo + (indicador if c == coluna_ativa else ""))

    def _renderiza_tabela_rebalanceamento(self):
        if self.df_rebalanceamento is None:
            return
        df = self.df_rebalanceamento

        filtro_fn = FILTROS_REBALANCEAMENTO.get(self.opcao_filtro_rebalanceamento.get())
        if filtro_fn is not None:
            df = filtro_fn(df)

        coluna, reverso = self._ordenacao_rebalanceamento
        chave_col = MAPA_ORDENACAO_REBALANCEAMENTO.get(coluna, coluna)
        if chave_col in df.columns:
            if df[chave_col].dtype == object:
                df = df.sort_values(by=chave_col, ascending=not reverso, key=lambda s: s.str.lower())
            else:
                df = df.sort_values(by=chave_col, ascending=not reverso)

        for item in self.tabela_rebalanceamento.get_children():
            self.tabela_rebalanceamento.delete(item)

        for _, row in df.iterrows():
            cotas = int(row["cotas_sugeridas"])
            if cotas > 0:
                tag = "comprar"
                texto_ajuste = f"Comprar {cotas} cota{'s' if cotas != 1 else ''} (R$ {row['valor_transacao']:.2f})"
            elif cotas < 0:
                tag = "vender"
                texto_ajuste = (f"Vender {abs(cotas)} cota{'s' if cotas != -1 else ''} "
                                 f"(R$ {abs(row['valor_transacao']):.2f})")
            else:
                tag = ""
                texto_ajuste = "Sem ajuste (< 1 cota)"
            self.tabela_rebalanceamento.insert("", "end", tags=(tag,), values=(
                row["ticker"],
                f"{row['peso_atual'] * 100:.1f}%",
                f"{row['peso_alvo'] * 100:.1f}%",
                f"{row['valor_atual']:.2f}",
                f"{row['preco_atual']:.2f}",
                texto_ajuste,
                row["ir_estimado"],
            ))

        self._atualiza_headers_ordenacao()

    def _estiliza_eixos(self, ax):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUTED)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(SURFACE_2)

    def _desenha_fronteira(self, r: dict):
        fig = self.fig_fronteira
        fig.clear()
        ax = fig.add_subplot(111)
        self._estiliza_eixos(ax)

        sim = r["simulacoes"]
        disp = ax.scatter(sim["volatilidade"] * 100, sim["retorno"] * 100, c=sim["sharpe"],
                           cmap=CMAP_SHARPE, s=8, alpha=0.55)
        cbar = fig.colorbar(disp, ax=ax)
        cbar.set_label("Indice de Sharpe", color=TEXT)
        cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(TEXT_MUTED)

        fronteira = r["fronteira"]
        if fronteira:
            vols = [p.volatilidade * 100 for p in fronteira]
            rets = [p.retorno_esperado * 100 for p in fronteira]
            ax.plot(vols, rets, color=TEXT, linewidth=2, label="Fronteira eficiente")

        for nome, chave in ESTRATEGIAS_REFERENCIA.items():
            p = r[chave]
            estilo = ESTILOS_MARCADOR_ESTRATEGIA[nome]
            ax.scatter([p.volatilidade * 100], [p.retorno_esperado * 100], color=estilo["color"],
                       marker=estilo["marker"], s=estilo["s"], label=nome, zorder=5,
                       edgecolors=INK, linewidths=0.5)

        atual = r["carteira_atual"]
        ax.scatter([atual.volatilidade * 100], [atual.retorno_esperado * 100], color="#4A90D9", marker="D",
                   s=100, label="Carteira atual", zorder=5, edgecolors=INK, linewidths=0.5)

        if self.escolhida is not None:
            ax.scatter([self.escolhida.volatilidade * 100], [self.escolhida.retorno_esperado * 100],
                       facecolors="none", edgecolors=HEARTWOOD_GLOW, marker="o", s=420, linewidths=2.5,
                       label="Alvo escolhido", zorder=6)

        ax.set_xlabel("Volatilidade anual (%)")
        ax.set_ylabel("Retorno esperado anual (%)")
        ax.set_title("Fronteira Eficiente de Markowitz - clique num ponto p/ escolher o alvo")
        ax.legend(loc="best", facecolor=SURFACE, edgecolor=SURFACE_2, labelcolor=TEXT, fontsize=8, ncol=2)
        fig.tight_layout()
        self.canvas_fronteira.draw()

    def _desenha_alocacao(self, r: dict):
        fig = self.fig_alocacao
        fig.clear()
        tickers = r["tickers"]
        cores = [cor_ativo(i) for i in range(len(tickers))]

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        def formata_pct(valor):
            return f"{valor:.1f}%" if valor >= 1.5 else ""

        pesos_atuais = r["carteira_atual"].pesos
        if pesos_atuais.sum() > 0:
            ax1.pie(pesos_atuais, autopct=formata_pct, colors=cores,
                    textprops={"color": INK, "fontweight": "bold"})
        else:
            ax1.set_facecolor(SURFACE)
            ax1.text(0.5, 0.5, "Sem posicao atual\n(carteira vazia - so aporte)",
                     ha="center", va="center", color=TEXT_MUTED, fontsize=11, wrap=True)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis("off")
        ax1.set_title("Carteira Atual", color=TEXT)

        alvo = self.escolhida if self.escolhida is not None else r["carteira_max_sharpe"]
        wedges2, _, _ = ax2.pie(alvo.pesos, autopct=formata_pct, colors=cores,
                                 textprops={"color": INK, "fontweight": "bold"})
        ax2.set_title("Alvo Escolhido na Fronteira", color=TEXT)

        fig.legend(wedges2, tickers, loc="lower center", ncol=min(len(tickers), 5),
                   facecolor=SURFACE, edgecolor=SURFACE_2, labelcolor=TEXT)
        fig.patch.set_facecolor(INK)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        self.canvas_alocacao.draw()

    def _desenha_risco(self, r: dict):
        retornos = r["retornos_diarios"]
        pesos = self.escolhida.pesos
        tickers = r["tickers"]

        rp = opt.retornos_portfolio(retornos, pesos)
        equity = (1 + rp).cumprod() * 100
        drawdown = equity / equity.cummax() - 1

        bm_ret = r["benchmark_precos"].pct_change().dropna()
        bm_equity = ((1 + bm_ret).cumprod() * 100).reindex(equity.index).ffill()

        taxa_livre = r["taxa_livre"]
        cdi_diario = (1 + taxa_livre) ** (1 / opt.DIAS_UTEIS_ANO) - 1
        cdi_equity = pd.Series([(1 + cdi_diario) ** i * 100 for i in range(len(equity))], index=equity.index)

        cov_anual = retornos.cov() * opt.DIAS_UTEIS_ANO
        contrib = opt.contribuicao_risco(pesos, cov_anual)

        vc = opt.var_cvar_historico(rp)
        total = sum(r["valores"]) + self._ler_aporte()
        self.label_var_cvar.configure(text=(
            f"VaR diario (95%): {vc['var_diario'] * 100:.2f}% (~R$ {vc['var_diario'] * total:.2f})   |   "
            f"CVaR diario (95%): {vc['cvar_diario'] * 100:.2f}% (~R$ {vc['cvar_diario'] * total:.2f})   -   "
            "perda esperada em dia ruim (media dos piores 5% dos dias no periodo historico baixado), "
            "calculado p/ o alvo escolhido na fronteira."))

        fig = self.fig_risco
        fig.clear()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        for ax in (ax1, ax2, ax3):
            self._estiliza_eixos(ax)

        ax1.plot(equity.index, equity.values, color=HEARTWOOD, linewidth=1.6, label="Carteira (alvo escolhido)")
        ax1.plot(bm_equity.index, bm_equity.values, color="#4A90D9", linewidth=1.2, label="Ibovespa (BOVA11)")
        ax1.plot(cdi_equity.index, cdi_equity.values, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
                  label="CDI aprox. (taxa livre de risco)")
        ax1.set_title("Desempenho acumulado - buy and hold (base 100)")
        ax1.legend(loc="upper left", fontsize=8, facecolor=SURFACE, edgecolor=SURFACE_2, labelcolor=TEXT)

        ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color=NEGATIVO, alpha=0.35)
        ax2.plot(drawdown.index, drawdown.values * 100, color=NEGATIVO, linewidth=1)
        ax2.set_title("Drawdown da carteira (%)")
        ax2.set_ylabel("%")

        cores = [cor_ativo(i) for i in range(len(tickers))]
        ax3.bar(range(len(tickers)), contrib * 100, color=cores)
        ax3.set_xticks(range(len(tickers)))
        ax3.set_xticklabels([t.replace(".SA", "") for t in tickers], rotation=45, ha="right")
        ax3.set_title("Contribuicao de risco por ativo (% da variancia total da carteira)")
        ax3.set_ylabel("%")

        fig.tight_layout()
        self.canvas_risco.draw()

    def _desenha_correlacao_e_retornos(self, r: dict):
        fig = self.fig_correlacao
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        self._estiliza_eixos(ax1)
        self._estiliza_eixos(ax2)

        corr = r["correlacao"]
        im = ax1.imshow(corr.values, cmap=CMAP_CORRELACAO, vmin=-1, vmax=1)
        ax1.set_xticks(range(len(corr.columns)))
        ax1.set_yticks(range(len(corr.columns)))
        ax1.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax1.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax1.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=8)
        ax1.set_title("Correlacao entre ativos")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(TEXT_MUTED)

        norm = r["precos_normalizados"]
        for i, col in enumerate(norm.columns):
            ax2.plot(norm.index, norm[col], label=col, linewidth=1.4, color=cor_ativo(i))
        ax2.set_title("Evolucao dos precos (base 100) - Buy and Hold")
        ax2.set_ylabel("Preco (base 100)")
        ax2.legend(loc="upper left", fontsize=8, facecolor=SURFACE, edgecolor=SURFACE_2, labelcolor=TEXT)

        fig.patch.set_facecolor(INK)
        fig.tight_layout()
        self.canvas_correlacao.draw()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
