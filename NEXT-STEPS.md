# Próximos Passos

## Próximos passos

Seguir esta ordem, sem desviar para CPU genérico:

1. manter o pipeline híbrido atual e recuperar a fidelidade do encoder `tiny_en_encoder_xint8.onnx` até transcrever `sample_hf_1.flac` corretamente
2. fazer o decoder Whisper `tiny_en_decoder_xint8.onnx` sair de `CPU only` e ganhar partição real no caminho DPU/DD
3. investigar FastFlowLM como caminho alternativo para LLM na NPU no Linux
4. manter baselines reproduzíveis para nao regredir (ResNet18 XINT8 + Whisper encoder XINT8)

## Passo 1 — CNN na NPU (cumprido)

- `ResNet18` quantizado por `Quark XINT8`: NPU 164 / CPU 2
- baseline reproduzível: `tools/run_resnet18_xint8_quark_probe.sh`

## Passo 1b — Transformer na NPU (cumprido)

- Whisper encoder (`amd/whisper-tiny-en-onnx-npu`) quantizado com `Quark XINT8`: NPU 416 / CPU 122
- primeiro transformer na NPU neste host Linux
- caminho que funciona: `Quark XINT8` + `vaip_config_npu_2_3.json` (DPU/DD)
- caminho VAIML (BF16) nao funciona nesta tree (falta `libvaip-pass_vaiml_partition.so`)
- baseline reproduzível: `tools/run_whisper_encoder_xint8_probe.sh`

## Passo 2 — Whisper completo na NPU (em progresso)

Estado atual:

- `tiny_en_decoder.onnx` ja foi baixado para `runtime/whisper/models`
- `tiny_en_decoder_xint8.onnx` ja foi gerado via `Quark XINT8`
- o decoder ja sobe e infere no host Linux via `tools/run_whisper_decoder_xint8_probe.sh`
- o relatorio atual do EP mostra `CPU 934` e nao gera `subgraphStat`
- `tools/run_whisper_full_hybrid.sh` ja valida audio real ponta a ponta com encoder na NPU + decoder FP32 em CPU
- o prefixo correto de `whisper-tiny.en` nesta trilha e `tokenizer.prefix_tokens`, isto e `[50257, 50362]`
- o melhor resultado atual no `sample_hf_1.flac` e `"I'm"`, melhor que `EOT` imediato mas ainda abaixo de transcricao util
- a fidelidade do encoder `XINT8` virou o principal gap do caminho hibrido
- `tools/run_whisper_npu_transcribe.py` segue existindo como scaffold no host

Objetivo:

- estabilizar a transcricao real no caminho hibrido atual
- provar particao real do decoder Whisper XINT8 na NPU
- depois unificar tudo em um pipeline completo: encoder + decoder + tokenizer + áudio real -> transcrição

Desafio técnico:

- o bloqueio principal do decoder ja nao e mais suporte basico a `2` inputs
- o problema do decoder continua sendo particao: ele segue inteiro em CPU mesmo apos `Quark XINT8`
- o problema do pipeline hibrido agora e fidelidade do encoder `XINT8`, nao mais prompt ou `EOT` imediato
- a quantização XINT8 do decoder deve preservar a autogressão
- a quantização XINT8 do encoder precisa preservar a semântica da fala, nao apenas o offload

Critério de sucesso:

- `sample_hf_1.flac` ou outro `.wav` real transcrevendo corretamente no caminho hibrido com encoder na NPU
- decoder com operadores na NPU (mesmo que parcial)
- transcrição real de `.wav` usando NPU para encoder e decoder

## Passo 3 — LLM na NPU

OGA no Linux está bloqueado por falta de binários proprietários AMD.

Caminhos possíveis:

1. `FastFlowLM` (runtime NPU-first, suporte Linux desde 2026-03-11) — menor risco
2. Build OGA v0.12+ from source com `--ort_home` apontando para AMD ORT 1.20.1
3. Exportar LLM pequeno para ONNX + XINT8 via Quark + VitisAI EP direto (experimental)

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

Provas fortes já obtidas:

- `resnet18_xint8_quark.onnx` inferindo com `NPU 164 / CPU 2` (CNN)
- `tiny_en_encoder_xint8.onnx` inferindo com `NPU 416 / CPU 122` (transformer)
- ambos marcam `Actually running on NPU`

Estado intermediario importante:

- `tiny_en_decoder_xint8.onnx` ja infere no host, mas ainda fica em `CPU 934` e sem `subgraphStat`

Próximas provas fortes a buscar:

- transcrição real Whisper usando NPU (encoder + decoder completos)
- LLM rodando na NPU via FastFlowLM ou VitisAI EP direto
