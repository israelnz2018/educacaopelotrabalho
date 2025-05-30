from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import pandas as pd

from leitura import ler_arquivo
from suporte import interpretar_coluna
from estatistica import ANALISES
from graficos import GRAFICOS
from agente import interpretar_analise  # ✅ Agente ativado

app = FastAPI()

@app.post("/analise")
async def analisar(
    request: Request,
    arquivo: UploadFile = File(None),
    ferramenta: str = Form(None),
    grafico: str = Form(None),
    coluna_y: str = Form(None),
    colunas_x: str | list[str] = Form(None)
):
    try:
        df = await ler_arquivo(arquivo)
        colunas_usadas = []

        if coluna_y and coluna_y.strip():
            colunas_usadas.append(interpretar_coluna(df, coluna_y))

        colunas_x_lista = []
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_x_lista = [x.strip() for x in colunas_x.split(",") if x.strip()]
            elif isinstance(colunas_x, list):
                colunas_x_lista = [x.strip() for x in colunas_x if isinstance(x, str) and x.strip()]

        for c in colunas_x_lista:
            colunas_usadas.append(interpretar_coluna(df, c))

        if not colunas_usadas:
            return JSONResponse(content={"erro": "Informe ao menos coluna_y ou colunas_x."}, status_code=422)

        for col in colunas_usadas:
            if col not in df.columns:
                return JSONResponse(content={"erro": f"Coluna '{col}' não encontrada no arquivo."}, status_code=400)

        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None
        explicacao_ia = None

        # ✅ Caso 1: análise estatística
        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)

            # ✅ Interpretação da análise pelo agente de IA
            explicacao_ia = interpretar_analise(resultado_texto)

        # ✅ Caso 2: gráfico isolado
        if grafico and grafico.strip():
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)
            if grafico.strip().startswith("boxplot"):
                imagem_grafico_isolado_base64 = funcao(df, colunas_usadas, coluna_y=interpretar_coluna(df, coluna_y))
            else:
                imagem_grafico_isolado_base64 = funcao(df, colunas_usadas)

        if not ferramenta and not grafico:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto or "",
            "explicacao_ia": explicacao_ia,
            "grafico_base64": imagem_analise_base64,
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"erro": "Erro interno ao processar a análise.", "detalhe": str(e)}, status_code=500)
