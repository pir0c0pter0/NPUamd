# Próximos Passos

## Próximos 3 passos

Seguir esta ordem, sem desviar para CPU genérico:

1. manter `ResNet18 XINT8` via `Quark` como baseline reproduzível do primeiro offload real no host Linux
2. fechar o runtime `OGA` no Linux e testar primeiro o modelo oficial `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
3. só depois testar `Qwen` da coleção AMD `OGA/NPU` e medir limites reais de contexto, TTFT, tokens/s, memória e fallback

## Passo 1

Objetivo:

- sair do estado atual em que o provider sobe, mas o grafo mínimo cai todo em CPU
- obter a primeira prova forte de offload real para a NPU

Critério de sucesso:

- pelo menos parte do grafo atribuída ao `VitisAIExecutionProvider`
- inferência reproduzível no host Linux

Estado atualizado desta rodada:

- o `probe_vitisai.c` já consegue rodar modelos ONNX reais, nao só `Identity`
- um CNN sintético com `2 Conv` já foi testado nativamente no host
- o `CNN` sintético continua em `CPU 8`, sem partição real
- um `ResNet18` com `QDQ` padrão do ORT ainda fica em `CPU 199`
- um `ResNet18` quantizado por `Quark XINT8` agora gera `NPU 164 / CPU 2`
- o relatório `runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json` marca `Actually running on NPU 1`
- o config BF16 mínimo do exemplo AMD falha nesta tree Linux local por plugin ausente: `libvaip-pass_vaiml_partition.so`

Leitura correta:

1. o passo 1 agora está cumprido
2. nem todo `QDQ` genérico basta; a quantização `Quark XINT8` foi decisiva
3. daqui em diante, `resnet18_xint8_quark.onnx` deve ser tratado como baseline para nao regredir

## Passo 2

Objetivo:

- validar LLM real no caminho certo da AMD, em Linux, sem depender de modelo genérico
- usar como primeira referência o fluxo oficial Linux da AMD para `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`

Critério de sucesso:

- `OGA` carregando no Linux
- primeira geração com modelo AMD de NPU ou hybrid

Estado atualizado desta rodada:

- o hardware local está na família suportada `STX/KRK`
- porém o runtime `OGA` ainda nao está materializado localmente em Linux
- a doc oficial atual da AMD ja tem uma pagina `Running LLM on Linux`
- essa pagina usa como referencia o modelo `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
- ela espera uma tree local `RYZEN_AI_INSTALLATION_PATH` com `deployment/lib/libonnx_custom_ops.so`, `libryzen_mm.so` e `model_benchmark`
- essa tree ainda nao existe neste host
- o clone cru do modelo `Phi-3.5` ainda vem em ponteiros `git-lfs` para os blobs reais

Conclusão prática:

- o passo 2 continua sendo meta do hub
- mas, antes de rodar `Phi-3.5`, é obrigatório materializar a instalacao Linux recente da AMD e os blobs reais do modelo

## Passo 3

Objetivo:

- só depois de `Phi-3.5` subir, trocar para `Qwen` AMD da coleção `OGA/NPU`
- medir o que a NPU entrega de verdade nesta máquina

Critério de sucesso:

- geração real com `Qwen` AMD no runtime correto
- números básicos de contexto, latência, throughput e memória

## O que nao fazer

### Nao insistir no host Python

Nao usar o host com:

- `Python 3.14`
- wheels recentes aleatórias de ONNX Runtime

Motivo:

- a stack AMD encontrada localmente gira em torno de `onnxruntime 1.20.1`
- a validação nativa atual já evita esse host Python via probe C + userspace compatível local

### Nao baixar aliases como se fossem binários reais

Nao confiar diretamente nestes nomes como binário final:

- `libonnxruntime.so`
- `libonnxruntime.so.1`
- `libxir.so`
- `libunilog.so`
- `libtarget-factory.so`
- `libvart-runner.so`
- `libvart-util.so`
- `libvart-mem-manager.so`
- `libxcompiler-xcompiler-core.so`
- `libxcompiler-xcompiler-core.so.3`

Regra:

- baixar o arquivo versionado real
- depois criar symlink local correto

### Nao usar `xrt-smi` atual como prova final

Hoje:

- `xrt-smi` ainda nao serve como validação final
- o estado continua sendo `0 devices found`

### Nao usar placeholder de `git-lfs` como se fosse modelo real

Nao tratar estes arquivos pequenos como checkpoint materializado:

- `RyzenAI-SW/CNN-examples/getting_started_resnet/bf16/models/resnet_trained_for_cifar10.pt`
- `RyzenAI-SW/CNN-examples/getting_started_resnet/int8/models/resnet_trained_for_cifar10.pt`

Motivo:

- o arquivo local atual é texto ASCII de `133` bytes
- ele é apenas ponteiro `git-lfs`
- o blob real ainda nao está no disco

### Nao tratar CNN sintético atual como prova de NPU

Hoje:

- o probe com CNN pequeno já executa com input/output reais
- o log continua mostrando `CPU 8`

Conclusão:

- esse teste é útil para depuração
- ele nao é prova de processamento real na NPU

### Nao presumir que `OGA` Linux já existe localmente

Nao assumir que:

- exemplos `OGA` no clone de `RyzenAI-SW` significam runtime pronto
- a pagina oficial `llm_linux` equivale a artefatos Linux ja baixados
- `Phi-3.5` oficial poderá rodar só com o que já está hoje no host

O que falta primeiro:

- a tree `RYZEN_AI_INSTALLATION_PATH` com `deployment/` e `LLM/examples/`
- `libonnx_custom_ops.so`, `libryzen_mm.so` e `model_benchmark`
- os blobs reais `git-lfs` do modelo `Phi-3.5`

### Nao presumir offload automático universal

Nao assumir que:

- `Ollama` vai usar NPU sozinho
- `Whisper` qualquer vai usar NPU sozinho
- qualquer processo IA vai cair automaticamente na NPU
- um Qwen simples em `transformers.js` prova uso de NPU
- `Qwen` genérico em ONNX ou GGUF equivale a `Qwen` AMD em `OGA`

O uso real depende de:

- runtime correto
- provider correto
- modelo compatível

## Meta de validação

A primeira prova forte já foi obtida com:

- `resnet18_xint8_quark.onnx` inferindo com `NPU 164 / CPU 2`
- `subgraphStat` marcando `Actually running on NPU 1`

As próximas provas fortes a buscar agora sao:

- um `Phi-3.5-mini-instruct` AMD rodando no Linux pelo caminho `OGA`
- um `Qwen` AMD rodando no Linux pelo caminho `OGA`
