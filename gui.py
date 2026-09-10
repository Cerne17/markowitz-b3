"""Dashboard customtkinter: alocacao otima de carteira B3 via Markowitz + Sharpe."""
import json
import queue
import threading
from tkinter import messagebox, ttk

import customtkinter as ctk
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
        self.escolhida: opt.Portfolio | None = None
        self.universo_ativos = dfx.carrega_universo_ativos()
        self.ativo_selecionado_explorar: dict | None = None

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
        painel = ctk.CTkFrame(self, width=340, fg_color=SURFACE)
        painel.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)
        painel.grid_propagate(False)

        ctk.CTkLabel(painel, text="Minha Carteira", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=14, pady=(14, 2))
        ctk.CTkLabel(painel, text="Valor em R$ investido em cada ativo (nao e quantidade de acoes/cotas)",
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

        self.frame_ativos = ctk.CTkScrollableFrame(painel, height=320, fg_color=SURFACE_2)
        self.frame_ativos.pack(fill="both", expand=True, padx=14, pady=(0, 6))

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
        self.tab_correlacao = self.tabview.add("Correlacao & Retornos")

        self._monta_tab_explorar()
        self._monta_tab_resumo()
        self._monta_canvas(self.tab_fronteira, "fronteira")
        self._monta_canvas(self.tab_alocacao, "alocacao")
        self._monta_canvas(self.tab_correlacao, "correlacao")

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

    def _on_muda_horizonte(self):
        if self.ativo_selecionado_explorar is not None:
            self._selecionar_ativo_explorar(self.ativo_selecionado_explorar)

    def _selecionar_ativo_explorar(self, ativo: dict):
        self.ativo_selecionado_explorar = ativo
        self.label_titulo_ativo.configure(text=f"{ativo['nome']} ({ativo['ticker']}) - {ativo['setor']}")
        self.label_info_ativo.configure(text="Carregando dados do Yahoo Finance...")
        self.label_stats_ativo.configure(text="")
        self.btn_adicionar_ativo.configure(state="disabled")
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
        if valor <= 0:
            messagebox.showerror("Valor invalido", "O valor deve ser maior que zero.")
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
        self.tab_resumo.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.card_atual = self._cria_card(self.tab_resumo, "Carteira Atual", 0)
        self.card_max_sharpe = self._cria_card(self.tab_resumo, "Referencia: Max Sharpe", 1)
        self.card_min_vol = self._cria_card(self.tab_resumo, "Referencia: Min Volatilidade", 2)
        self.card_escolhida = self._cria_card(self.tab_resumo, "Alvo Escolhido na Fronteira", 3,
                                               destaque=True)

        self.label_titulo_rebalanceamento = ctk.CTkLabel(
            self.tab_resumo,
            text="Sugestao de rebalanceamento (clique num ponto da Fronteira Eficiente p/ mudar o alvo)",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT, wraplength=900, justify="left")
        self.label_titulo_rebalanceamento.grid(row=1, column=0, columnspan=4, sticky="w", padx=16, pady=(18, 4))

        estilo_tv = ttk.Style()
        estilo_tv.theme_use("default")
        estilo_tv.configure("Treeview", background=SURFACE_2, fieldbackground=SURFACE_2,
                             foreground=TEXT, rowheight=26, borderwidth=0)
        estilo_tv.configure("Treeview.Heading", background=INK, foreground=TEXT)
        estilo_tv.map("Treeview", background=[("selected", HEARTWOOD)], foreground=[("selected", INK)])

        colunas = ("ticker", "peso_atual", "peso_alvo", "valor_atual", "valor_alvo", "ajuste")
        self.tabela_rebalanceamento = ttk.Treeview(self.tab_resumo, columns=colunas, show="headings", height=8)
        titulos = {"ticker": "Ticker", "peso_atual": "Peso atual", "peso_alvo": "Peso alvo",
                   "valor_atual": "Valor atual (R$)", "valor_alvo": "Valor alvo (R$)", "ajuste": "Ajuste (R$)"}
        for c in colunas:
            self.tabela_rebalanceamento.heading(c, text=titulos[c])
            self.tabela_rebalanceamento.column(c, anchor="center", width=140)
        self.tabela_rebalanceamento.tag_configure("comprar", foreground=POSITIVO)
        self.tabela_rebalanceamento.tag_configure("vender", foreground=NEGATIVO)
        self.tabela_rebalanceamento.grid(row=2, column=0, columnspan=4, sticky="nswe", padx=16, pady=6)
        self.tab_resumo.grid_rowconfigure(2, weight=1)

    def _cria_card(self, master, titulo, coluna, destaque=False):
        card = ctk.CTkFrame(master, fg_color=SURFACE_2,
                             border_width=2 if destaque else 0,
                             border_color=HEARTWOOD if destaque else SURFACE_2)
        card.grid(row=0, column=coluna, sticky="nswe", padx=10, pady=14)
        ctk.CTkLabel(card, text=titulo, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=HEARTWOOD if destaque else TEXT).pack(pady=(12, 6))
        label_valores = ctk.CTkLabel(card, text="--", justify="left", font=ctk.CTkFont(size=13), text_color=TEXT)
        label_valores.pack(padx=16, pady=(0, 14))
        card.label_valores = label_valores
        return card

    def _monta_canvas(self, tab, chave):
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=INK)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        setattr(self, f"fig_{chave}", fig)
        setattr(self, f"canvas_{chave}", canvas)

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
            if valor <= 0:
                raise ValueError(f"Valor investido invalido para {ticker}.")
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

            precos = dfx.baixar_precos(tickers, cfg["data_inicio"])
            precos = precos[tickers]
            retornos = dfx.calcula_retornos_diarios(precos)
            media_anual, cov_anual = opt.estatisticas_anuais(retornos)
            taxa_livre = cfg["taxa_livre_risco_anual"]
            peso_maximo = cfg.get("peso_maximo_por_ativo", 1.0)

            pesos_atuais = opt.pesos_atuais(valores)
            carteira_atual = opt.desempenho_portfolio(pesos_atuais, media_anual, cov_anual, taxa_livre)
            carteira_max_sharpe = opt.otimiza_max_sharpe(media_anual, cov_anual, taxa_livre, peso_maximo)
            carteira_min_vol = opt.otimiza_min_volatilidade(media_anual, cov_anual, taxa_livre, peso_maximo)
            fronteira = opt.fronteira_eficiente(media_anual, cov_anual, taxa_livre, peso_maximo_por_ativo=peso_maximo)
            simulacoes = opt.simula_portfolios_aleatorios(
                media_anual, cov_anual, taxa_livre, n_simulacoes=cfg["num_portfolios_simulados"])
            correlacao = retornos.corr()
            precos_normalizados = precos / precos.iloc[0] * 100

            self.fila.put(("ok", {
                "tickers": tickers,
                "valores": valores,
                "carteira_atual": carteira_atual,
                "carteira_max_sharpe": carteira_max_sharpe,
                "carteira_min_vol": carteira_min_vol,
                "fronteira": fronteira,
                "simulacoes": simulacoes,
                "correlacao": correlacao,
                "precos_normalizados": precos_normalizados,
            }))
        except Exception as e:
            self.fila.put(("erro", str(e)))

    def _processa_fila(self):
        try:
            tipo, payload = self.fila.get_nowait()
        except queue.Empty:
            pass
        else:
            if tipo == "erro":
                self.btn_calcular.configure(state="normal", text="Calcular alocacao otima")
                self.label_status.configure(text=f"Erro: {payload}", text_color=NEGATIVO)
                messagebox.showerror("Erro ao calcular", payload)
            elif tipo == "ok":
                self.btn_calcular.configure(state="normal", text="Calcular alocacao otima")
                self.label_status.configure(text="Calculo concluido. Clique na fronteira p/ escolher o alvo.",
                                             text_color=POSITIVO)
                self._atualiza_ui(payload)
            elif tipo == "ativo_erro":
                self.label_info_ativo.configure(text=f"Erro ao carregar: {payload}")
            elif tipo == "ativo_ok":
                self._atualiza_analise_ativo(payload)
        self.after(150, self._processa_fila)

    # ---------- atualizacao visual ----------
    def _atualiza_ui(self, r: dict):
        self.resultado = r
        self.pontos_fronteira = r["fronteira"]
        self.escolhida = r["carteira_max_sharpe"]  # alvo padrao ate o usuario clicar na fronteira

        self._atualiza_cards(r)
        self._atualiza_escolhida()
        self._desenha_alocacao(r)
        self._desenha_correlacao_e_retornos(r)

    def _atualiza_cards(self, r: dict):
        def texto(p, pesos_tickers=None):
            base = (f"Retorno esperado: {p.retorno_esperado * 100:.2f}% a.a.\n"
                    f"Volatilidade: {p.volatilidade * 100:.2f}% a.a.\n"
                    f"Indice de Sharpe: {p.sharpe:.3f}")
            if pesos_tickers:
                pesos_txt = "\n".join(f"  {t}: {w * 100:.1f}%" for t, w in pesos_tickers)
                base += f"\n\nPesos:\n{pesos_txt}"
            return base

        tickers = r["tickers"]
        self.card_atual.label_valores.configure(text=texto(r["carteira_atual"]))
        self.card_max_sharpe.label_valores.configure(
            text=texto(r["carteira_max_sharpe"], list(zip(tickers, r["carteira_max_sharpe"].pesos))))
        self.card_min_vol.label_valores.configure(
            text=texto(r["carteira_min_vol"], list(zip(tickers, r["carteira_min_vol"].pesos))))
        self._texto_portfolio = texto  # reutilizado por _atualiza_escolhida

    # ---------- selecao interativa na fronteira ----------
    def _on_click_fronteira(self, event):
        if event.inaxes is None or not self.pontos_fronteira or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        melhor_idx, melhor_dist = 0, float("inf")
        for i, p in enumerate(self.pontos_fronteira):
            dist = (p.volatilidade * 100 - x) ** 2 + (p.retorno_esperado * 100 - y) ** 2
            if dist < melhor_dist:
                melhor_idx, melhor_dist = i, dist
        self.escolhida = self.pontos_fronteira[melhor_idx]
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

    def _recalcula_tabela_rebalanceamento(self):
        if self.resultado is None or self.escolhida is None:
            return
        r = self.resultado
        aporte = self._ler_aporte()
        total = sum(r["valores"]) + aporte

        rebalanceamento = opt.sugestao_rebalanceamento(r["tickers"], r["valores"], self.escolhida.pesos, aporte)
        self._atualiza_tabela_rebalanceamento(rebalanceamento)

        self.label_titulo_rebalanceamento.configure(
            text=(f"Sugestao de rebalanceamento - carteira atual (R$ {sum(r['valores']):.2f}) "
                  f"+ aporte (R$ {aporte:.2f}) = total R$ {total:.2f}. "
                  "Clique num ponto da Fronteira Eficiente p/ mudar o alvo."))

    def _atualiza_tabela_rebalanceamento(self, df):
        for item in self.tabela_rebalanceamento.get_children():
            self.tabela_rebalanceamento.delete(item)
        for _, row in df.iterrows():
            tag = "comprar" if row["ajuste"] > 0 else ("vender" if row["ajuste"] < 0 else "")
            self.tabela_rebalanceamento.insert("", "end", tags=(tag,), values=(
                row["ticker"],
                f"{row['peso_atual'] * 100:.1f}%",
                f"{row['peso_alvo'] * 100:.1f}%",
                f"{row['valor_atual']:.2f}",
                f"{row['valor_alvo']:.2f}",
                f"{row['ajuste']:+.2f}",
            ))

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

        ms = r["carteira_max_sharpe"]
        mv = r["carteira_min_vol"]
        atual = r["carteira_atual"]
        ax.scatter([ms.volatilidade * 100], [ms.retorno_esperado * 100], color=HEARTWOOD, marker="*",
                   s=240, label="Max Sharpe", zorder=5, edgecolors=INK, linewidths=0.5)
        ax.scatter([mv.volatilidade * 100], [mv.retorno_esperado * 100], color=SAPWOOD, marker="*",
                   s=200, label="Min Volatilidade", zorder=5, edgecolors=INK, linewidths=0.5)
        ax.scatter([atual.volatilidade * 100], [atual.retorno_esperado * 100], color="#4A90D9", marker="D",
                   s=100, label="Carteira atual", zorder=5, edgecolors=INK, linewidths=0.5)

        if self.escolhida is not None:
            ax.scatter([self.escolhida.volatilidade * 100], [self.escolhida.retorno_esperado * 100],
                       facecolors="none", edgecolors=HEARTWOOD_GLOW, marker="o", s=420, linewidths=2.5,
                       label="Alvo escolhido", zorder=6)

        ax.set_xlabel("Volatilidade anual (%)")
        ax.set_ylabel("Retorno esperado anual (%)")
        ax.set_title("Fronteira Eficiente de Markowitz - clique num ponto p/ escolher o alvo")
        ax.legend(loc="best", facecolor=SURFACE, edgecolor=SURFACE_2, labelcolor=TEXT)
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

        wedges1, _, _ = ax1.pie(r["carteira_atual"].pesos, autopct=formata_pct, colors=cores,
                                 textprops={"color": INK, "fontweight": "bold"})
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
