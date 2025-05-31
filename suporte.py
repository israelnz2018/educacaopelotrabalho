def interpretar_coluna(df, nome_coluna):
    """
    Interpreta a coluna informada:
    - Se for uma letra (ex: 'A'), retorna o nome correspondente da coluna no DataFrame.
    - Se for o nome real da coluna (ex: 'Eficiencia (y)'), retorna diretamente se existir.
    """
    nome_coluna = nome_coluna.strip()

    # Caso seja letra (ex: 'A', 'B'...)
    if len(nome_coluna) == 1 and nome_coluna.upper().isalpha():
        index = ord(nome_coluna.upper()) - ord('A')
        if 0 <= index < len(df.columns):
            return df.columns[index]
        else:
            raise ValueError(f"Letra '{nome_coluna}' está fora do intervalo de colunas.")

    # Caso seja nome real da coluna
    if nome_coluna in df.columns:
        return nome_coluna

    raise ValueError(f"Coluna '{nome_coluna}' é inválida.")





