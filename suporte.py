def interpretar_coluna(df, nome_coluna):
    """
    Interpreta a(s) coluna(s) informada(s) como letras (ex: 'A', 'B', 'C') e retorna o(s) nome(s) real(is) da coluna no DataFrame.
    Suporta letras únicas, listas de letras e strings separadas por vírgula.
    """
    try:
        if isinstance(nome_coluna, list):
            letras = nome_coluna
        elif isinstance(nome_coluna, str):
            letras = [x.strip().upper() for x in nome_coluna.split(",") if x.strip()]
        else:
            raise ValueError

        nomes_reais = []
        for letra in letras:
            if len(letra) != 1 or not letra.isalpha():
                raise ValueError(f"Coluna '{letra}' é inválida.")
            index = ord(letra) - ord('A')
            if index < 0 or index >= len(df.columns):
                raise ValueError(f"Coluna '{letra}' está fora do intervalo.")
            nomes_reais.append(df.columns[index])

        return nomes_reais if len(nomes_reais) > 1 else nomes_reais[0]

    except Exception as e:
        raise ValueError(f"Coluna '{nome_coluna}' é inválida.")




