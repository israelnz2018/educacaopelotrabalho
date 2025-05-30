import os

try:
    import openai
except ImportError:
    openai = None

def interpretar_analise(analise_texto):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_API_KEY or not openai:
        return "O agente de IA ainda não está ativado. Configure a chave OPENAI_API_KEY para usar esta função."

    openai.api_key = OPENAI_API_KEY

    prompt = f"""
Você é um especialista em estatística com habilidade de explicar análises de forma clara e útil. Analise o seguinte resultado estatístico e explique o que ele significa, de forma prática e objetiva:

\"\"\"{analise_texto}\"\"\"

Evite repetir números. Foque no significado e nas implicações dos resultados.
"""

    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )
        return resposta.choices[0].message["content"].strip()
    except Exception as e:
        return f"Erro ao acessar o agente de IA: {str(e)}"

