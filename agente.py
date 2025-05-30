import openai

# Configure sua chave da OpenAI (ou use variável de ambiente)
openai.api_key = "SUA_CHAVE_AQUI"  # Substitua ou remova se usar variável externa

def interpretar_analise(texto_analise: str) -> str:
    prompt = f"""
Você é um especialista em estatística. Interprete de forma clara e profissional a seguinte análise estatística para um usuário leigo:

{texto_analise}

Explique o que os valores significam, se há algo relevante, e o que isso poderia indicar na prática.
"""

    resposta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um especialista em estatística que explica análises de forma clara para o público."},
            {"role": "user", "content": prompt}
        ]
    )

    return resposta.choices[0].message.content.strip()
