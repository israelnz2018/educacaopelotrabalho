import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import os
import io
from io import BytesIO


from estilo import aplicar_estilo_minitab

def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

def grafico_dispersao(df, colunas):
    if len(colunas) < 2:
        raise ValueError("Gráfico de dispersão requer exatamente duas colunas.")
    plt.figure(figsize=(8, 6))
    aplicar_estilo_minitab()
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def grafico_boxplot_simples(df, colunas, coluna_y=None):
    if not coluna_y:
        raise ValueError("Para o boxplot simples, a coluna Y (numérica) é obrigatória.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    df_box = pd.DataFrame({coluna_y: y, "grupo": "A"})

    plt.figure(figsize=(6, 6))
    aplicar_estilo_minitab()
    sns.boxplot(data=df_box, x="grupo", y=coluna_y, color="#89CFF0", width=0.3)
    sns.pointplot(data=df_box, x="grupo", y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.2, errwidth=0)
    plt.xlabel("")
    plt.ylabel(coluna_y)
    plt.title("Boxplot Simples com Média (losango)")
    return salvar_grafico()

def grafico_pareto(df, colunas):
    if len(colunas) != 1:
        raise ValueError("O gráfico de Pareto requer exatamente uma coluna categórica.")

    coluna = colunas[0]
    contagem = df[coluna].value_counts().sort_values(ascending=False)
    porcentagem = contagem / contagem.sum() * 100
    acumulado = porcentagem.cumsum()

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    ax = sns.barplot(x=contagem.index, y=contagem.values, color="#89CFF0")
    ax.set_ylabel("Frequência")
    ax.set_xlabel(coluna)
    ax.set_title("Gráfico de Pareto")

    ax2 = ax.twinx()
    ax2.plot(contagem.index, acumulado.values, color="red", marker="o", linewidth=2)
    ax2.set_ylabel("Acumulado (%)")
    ax2.set_ylim(0, 110)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return salvar_grafico()

def grafico_boxplot_multiplo(df, colunas, coluna_y=None):
    if not coluna_y or not coluna_y.strip():
        raise ValueError("Você deve selecionar uma coluna Y com valores numéricos para o boxplot múltiplo.")

    coluna_y = coluna_y.strip()
    if coluna_y.startswith("Unnamed") or coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não tem título válido ou não foi encontrada.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    if y.dropna().shape[0] < 2:
        raise ValueError("A coluna Y deve conter ao menos dois valores numéricos válidos.")

    colunas = [c.strip() for c in colunas if c.strip() and c.strip() != coluna_y and not c.strip().startswith("Unnamed")]
    if not colunas:
        raise ValueError("Nenhuma coluna X válida foi selecionada para o agrupamento.")

    x_col = colunas[0]
    if x_col not in df.columns:
        raise ValueError(f"A coluna X '{x_col}' não foi encontrada no arquivo.")

    grupo = df[x_col].astype(str)
    df_plot = pd.DataFrame({coluna_y: y, x_col: grupo}).dropna()

    if df_plot.empty:
        raise ValueError("Os dados da coluna Y e do grupo X selecionado não têm valores válidos simultaneamente.")

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    sns.boxplot(data=df_plot, x=x_col, y=coluna_y, color="#89CFF0")
    sns.pointplot(data=df_plot, x=x_col, y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.1, errwidth=0)

    plt.title(f"Boxplot Múltiplo por '{x_col}'")
    plt.xlabel(x_col)
    plt.ylabel(coluna_y)
    plt.xticks(rotation=45)

    return salvar_grafico()

def grafico_histograma_simples(df, colunas, coluna_y=None):
    if not colunas or len(colunas) == 0:
        raise ValueError("Você deve selecionar uma coluna Y com dados numéricos para o histograma simples.")

    coluna_y = colunas[0]

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()

    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    plt.figure(figsize=(8, 6))
    aplicar_estilo_minitab()
    sns.histplot(y, kde=True, color="#89CFF0", edgecolor="black")
    plt.xlabel(coluna_y)
    plt.ylabel("Frequência")
    plt.title("Histograma Simples com Curva de Densidade")

    return salvar_grafico()

def grafico_histograma_multiplo(df, colunas, coluna_y=None):
    if not coluna_y or not coluna_y.strip():
        raise ValueError("Você deve selecionar uma coluna Y com dados numéricos.")
    if not colunas or len(colunas) == 0:
        raise ValueError("Você deve selecionar uma coluna X com os grupos.")

    coluna_y = coluna_y.strip()
    coluna_x = colunas[0].strip()

    # 🚨 Correção importante: garantir que coluna_y != coluna_x
    if coluna_y == coluna_x:
        raise ValueError("A coluna Y e a coluna X devem ser diferentes.")

    if coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não foi encontrada no arquivo.")
    if coluna_x not in df.columns:
        raise ValueError(f"A coluna X '{coluna_x}' não foi encontrada no arquivo.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    grupo = df[coluna_x].astype(str)

    df_plot = pd.DataFrame({coluna_y: y, coluna_x: grupo}).dropna()

    if df_plot.empty:
        raise ValueError("Os dados das colunas selecionadas não têm valores válidos.")

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    cores = sns.color_palette("tab10", n_colors=df_plot[coluna_x].nunique())

    for i, (nome_grupo, dados_grupo) in enumerate(df_plot.groupby(coluna_x)):
        dados_y = pd.to_numeric(dados_grupo[coluna_y], errors="coerce").dropna()

        sns.histplot(
            x=dados_y,
            kde=True,
            stat="density",
            element="step",
            fill=True,
            label=str(nome_grupo),
            color=cores[i],
            alpha=0.4,
            edgecolor="black"
        )
        sns.kdeplot(
            x=dados_y,
            color=cores[i],
            linewidth=2,
            alpha=0.9
        )

    plt.title(f"Histograma Múltiplo de '{coluna_y}' por '{coluna_x}'")
    plt.xlabel(coluna_y)
    plt.ylabel("Densidade")
    plt.legend(title=coluna_x)
    return salvar_grafico()

def grafico_barras_simples(df, colunas_usadas):
    if len(colunas_usadas) != 1:
        raise ValueError("Selecione exatamente uma coluna para o Gráfico de Barras Simples.")

    nome_coluna = colunas_usadas[0]
    serie = df[nome_coluna].dropna()
    contagem = serie.value_counts().sort_index()

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    contagem.plot(kind='bar', color='skyblue', ax=ax)

    ax.set_title(f"Gráfico de Barras - {nome_coluna}")
    ax.set_xlabel(nome_coluna)
    ax.set_ylabel("Frequência")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64


def grafico_barras_agrupado(df, colunas_x, coluna_y):
    if not colunas_x or not isinstance(colunas_x, list):
        raise ValueError("Esperada uma lista com exatamente uma coluna X.")

    coluna_x = colunas_x[0]

    if coluna_x not in df.columns:
        raise ValueError(f"A coluna X '{coluna_x}' não foi encontrada no DataFrame.")
    if coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não foi encontrada no DataFrame.")

    aplicar_estilo_minitab()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=coluna_x, hue=coluna_y, ax=ax)

    ax.set_title(f'Gráfico de Barras Agrupado: {coluna_x} por {coluna_y}')
    ax.set_xlabel(coluna_x)
    ax.set_ylabel("Contagem")
    ax.legend(title=coluna_y)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64

GRAFICOS = {
    "scatter": grafico_dispersao,
    "boxplot_simples": grafico_boxplot_simples,
    "grafico_pareto": grafico_pareto,
    "boxplot_multiplo": grafico_boxplot_multiplo,
    "histograma_simples": grafico_histograma_simples,
    "histograma_multiplo": grafico_histograma_multiplo,
    "grafico_barras_simples": grafico_barras_simples,
    "Grafico_barras_agrupado": grafico_barras_agrupado
    
}

