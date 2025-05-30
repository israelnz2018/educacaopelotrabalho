def interpretar_coluna(df, nome_coluna):
    """
    Interpreta a coluna informada (como 'A', 'B', 'C') e retorna o nome real da coluna no DataFrame.
    """
    try:
        nome_coluna = nome_coluna.strip().upper()
        if len(nome_coluna) != 1 or not nome_coluna.isalpha():
            raise ValueError
        index = ord(nome_coluna) - ord('A')
        return df.columns[index]
    except Exception:
        raise ValueError(f"Coluna '{nome_coluna}' é inválida.")



