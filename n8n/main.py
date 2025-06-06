from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

@app.get("/healthz")
def healthcheck():
    return JSONResponse({"status": "ok"})

# monta todos os arquivos de n8n/ como "/html_app/..."
app.mount("/html_app", StaticFiles(directory=os.path.dirname(__file__)), name="html_app")
templates = Jinja2Templates(directory=os.path.dirname(__file__))

@app.get("/", response_class=HTMLResponse)
async def raiz(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

        nome_coluna_y = None
        if coluna_y and coluna_y.strip():
            nome_coluna_y = interpretar_coluna(df, coluna_y.strip())
            colunas_usadas.append(nome_coluna_y)

        colunas_x_lista = []
        if colunas_x:
            if isinstance(colunas_x, str):
                colunas_x_lista = [x.strip() for x in colunas_x.split(",") if x.strip()]
            elif isinstance(colunas_x, list):
                for item in colunas_x:
                    colunas_x_lista.extend([x.strip() for x in item.split(",") if x.strip()])

            for letra in colunas_x_lista:
                nome_coluna = interpretar_coluna(df, letra)
                colunas_usadas.append(nome_coluna)

        if not colunas_usadas:
            return JSONResponse(content={"erro": "Informe ao menos coluna_y ou colunas_x."}, status_code=422)

        for col in colunas_usadas:
            if col not in df.columns:
                return JSONResponse(content={"erro": f"Coluna '{col}' não encontrada no arquivo."}, status_code=400)

        resultado_texto = None
        imagem_analise_base64 = None
        imagem_grafico_isolado_base64 = None
        explicacao_ia = None

        if ferramenta and ferramenta.strip():
            funcao = ANALISES.get(ferramenta.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Análise estatística desconhecida."}, status_code=400)
            resultado_texto, imagem_analise_base64 = funcao(df, colunas_usadas)
            explicacao_ia = interpretar_analise(resultado_texto)

        if grafico and grafico.strip():
            funcao = GRAFICOS.get(grafico.strip())
            if not funcao:
                return JSONResponse(content={"erro": "Gráfico desconhecido."}, status_code=400)

            imagem_grafico_isolado_base64 = funcao(
                df,
                colunas_usadas,
                coluna_y=nome_coluna_y
            )

        if not ferramenta and not grafico:
            return JSONResponse(content={"erro": "Nenhuma ferramenta selecionada."}, status_code=400)

        return {
            "analise": resultado_texto or "",
            "explicacao_ia": explicacao_ia,
            "grafico_base64": imagem_analise_base64 if isinstance(imagem_analise_base64, str) else (imagem_analise_base64 or []),
            "grafico_isolado_base64": imagem_grafico_isolado_base64,
            "colunas_utilizadas": colunas_usadas
        }

    except ValueError as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)

    except Exception as e:
        tb = traceback.format_exc()
        print("🔴 ERRO COMPLETO:\n", tb)
        return JSONResponse(
            content={
                "erro": "Erro interno ao processar a análise.",
                "detalhe": str(e),
                "traceback": tb
            },
            status_code=500
        )


