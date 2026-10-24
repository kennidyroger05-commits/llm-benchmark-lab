# llm-benchmark-lab

Uma ferramenta Python simples e reproduzível para benchmarks empíricos de modelos de linguagem. O projeto está na etapa inicial: por enquanto, apenas valida a conexão com a API da Groq e lista os modelos disponíveis.

## Requisitos

- Python 3.9 ou superior
- Uma chave da API da Groq

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Configuração

Copie `.env.example` para `.env` e substitua `sua_chave_aqui` pela sua chave:

```env
GROQ_API_KEY=sua_chave_aqui
```

Mantenha `.env` local; ele está listado no `.gitignore`.

## Teste de conexão

Com o ambiente virtual ativado e o arquivo `.env` configurado, execute:

```bash
python run_benchmark.py
```

O programa informa se a conexão funcionou e lista os IDs de modelos retornados pela API. Nenhum benchmark é executado nesta etapa.
