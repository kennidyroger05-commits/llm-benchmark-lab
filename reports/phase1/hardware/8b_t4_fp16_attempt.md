# Tentativa do Llama 3.1 8B FP16 na Tesla T4

## Configuração

- **Modelo:** `meta-llama/Llama-3.1-8B-Instruct`
- **Fase:** `phase1`
- **Condição:** `raw`
- **dtype:** `float16`
- **device_map:** `auto`
- **GPU:** Tesla T4
- **VRAM disponível observada:** aproximadamente 14.56 GB

## Observação

Os pesos do modelo foram baixados e carregados. Durante o carregamento, Transformers/Accelerate informou que parâmetros foram offloaded para a CPU. A inferência tornou-se impraticavelmente lenta; após mais de 16 minutos, o benchmark ainda não havia concluído. A execução foi interrompida manualmente e terminou com `KeyboardInterrupt`.

## Resultado e interpretação

Nenhum arquivo JSON ou CSV oficial foi produzido. Esta tentativa não deve ser contabilizada como resultado de acurácia. Deve ser considerada evidência de limitação de hardware e de viabilidade do modelo 8B em FP16 na Tesla T4.
