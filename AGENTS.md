# Agentes

## Agente principal

Este trabalho foi conduzido pelo agente principal `Codex`.

Função:

- investigar o estado da NPU AMD no Linux
- montar a stack Linux local da AMD
- validar o runtime em container
- registrar o estado técnico do ambiente

## Subagentes

Até este ponto, nenhum subagente separado foi usado.

Estado:

- todo o trabalho foi feito diretamente pelo agente principal

## Como retomar depois

Ao reabrir e continuar, o agente deve começar por:

1. ler [README.md](/var/home/mariostjr/Documents/hubs/NPUamd/README.md)
2. ler [STATUS.md](/var/home/mariostjr/Documents/hubs/NPUamd/STATUS.md)
3. ler [NEXT-STEPS.md](/var/home/mariostjr/Documents/hubs/NPUamd/NEXT-STEPS.md)
4. usar [COMMANDS.md](/var/home/mariostjr/Documents/hubs/NPUamd/COMMANDS.md) como base operacional

## Escopo do próximo agente

Se um novo agente assumir este hub, o foco inicial deve ser:

1. manter `resnet18_xint8_quark.onnx` como baseline reproduzível do primeiro offload real em `VitisAIExecutionProvider`
2. fechar o runtime `OGA` no Linux usando primeiro `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
3. só depois testar `Qwen` da coleção AMD `OGA/NPU` e medir limites reais

## O que o próximo agente nao deve fazer

- nao voltar para Windows
- nao insistir no host `Python 3.14`
- nao tratar `xrt-smi` atual como prova final
- nao rebaixar o problema para “hardware nao detectado”, porque isso já foi resolvido
- nao gastar tempo com baseline CPU como se isso fosse prova de NPU
- nao usar `transformers.js`, `GGUF`, `Ollama` ou `Qwen` genérico como se fossem caminho oficial da NPU AMD
