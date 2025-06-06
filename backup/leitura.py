
import pandas as pd
import io
import os

from fastapi.responses import JSONResponse

async def ler_arquivo(arquivo):
    if arquivo and arquivo.filename.endswith(".xlsx"):
        file_bytes = await arquivo.read()
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        df.columns = df.columns.str.strip()
        return df
    else:
        raise ValueError("Envie um arquivo Excel (.xlsx) válido.")
