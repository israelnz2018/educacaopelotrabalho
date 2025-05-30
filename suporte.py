def interpretar_coluna(df, nome_coluna):
    """
    Interpreta a coluna informada (como 'A', 'B', 'C') e retorna o nome real da coluna no DataFrame.
    """
    try:
        index = ord(nome_coluna.upper()) - ord('A')
        return df.columns[index]
    except Exception:
        raise ValueError(f"Coluna '{nome_coluna}' é inválida.")

