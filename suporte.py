def interpretar_coluna(df, valor):
    valor = valor.strip()
    if len(valor) == 1 and valor.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(valor.upper()) - ord("A")
        if idx < len(df.columns):
            return df.columns[idx]
        else:
            raise ValueError(f"Coluna na posição '{valor}' não existe no arquivo. Arquivo tem apenas {len(df.columns)} colunas.")
    return valor
