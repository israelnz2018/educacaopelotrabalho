import matplotlib.pyplot as plt

def aplicar_estilo_minitab():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,        # Título do gráfico
        "axes.labelsize": 14,        # Nome dos eixos X e Y
        "xtick.labelsize": 12,       # Números no eixo X
        "ytick.labelsize": 12,       # Números no eixo Y
        "legend.fontsize": 12,       # Tamanho do texto da legenda
        "font.size": 13,             # Tamanho global da fonte
        "font.family": "Arial",
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.7,
        "legend.frameon": False
    })
    plt.grid(True)
