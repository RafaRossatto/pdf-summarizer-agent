import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Verificar se as variáveis existem
MY_KEY = os.getenv("MY_KEY")
URL = os.getenv("BASE_URL")

if not MY_KEY:
    print("❌ Erro: MY_KEY não encontrada no arquivo .env")
    exit(1)
if not URL:
    print("❌ Erro: BASE_URL não encontrada no arquivo .env")
    exit(1)

print(f"🔗 Conectando à URL: {URL}")
print(f"🔑 Usando chave: {MY_KEY[:10]}...")  # Mostra apenas início da chave

try:
    client = OpenAI(api_key=MY_KEY, base_url=URL)
    print("✅ Cliente OpenAI inicializado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao inicializar cliente: {e}")
    exit(1)

def perguntar_sobre_texto(pergunta):
    try:
        response = client.chat.completions.create(
            model="stepfun/step-3.5-flash:free",
            messages=[
                {"role": "system", "content": "Você é um assistente que responde em português de forma sucinta."},
                {"role": "user", "content": f"Pergunta: {pergunta}"}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro: {e}"

def main():
    print()  # Linha em branco
    pergunta = input("📝 Sua pergunta: ")
    resposta = perguntar_sobre_texto(pergunta)
    print(f"\n💬 Resposta: {resposta}")

if __name__ == "__main__":
    main()