# Próximos Passos

## Próximos passos

Seguir esta ordem, sem desviar para CPU genérico:

1. fechar `OGA Linux` com `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
2. subir `Qwen3-14B-onnx-ryzenai-1.7-hybrid` no caminho híbrido oficial AMD
3. abrir a frente separada de `Qwen3` grande em GPU, começando pela materialização da stack `ROCm/PyTorch`
4. manter baselines reproduzíveis para nao regredir (`ResNet18 XINT8` + `Whisper encoder XINT8`)

## Passo 1 — CNN na NPU (cumprido)

- `ResNet18` quantizado por `Quark XINT8`: NPU 164 / CPU 2
- baseline reproduzível: `tools/run_resnet18_xint8_quark_probe.sh`

## Passo 1b — Transformer na NPU (cumprido)

- Whisper encoder (`amd/whisper-tiny-en-onnx-npu`) quantizado com `Quark XINT8`: NPU 416 / CPU 122
- primeiro transformer na NPU neste host Linux
- caminho que funciona: `Quark XINT8` + `vaip_config_npu_2_3.json` (DPU/DD)
- caminho VAIML (BF16) nao funciona nesta tree (falta `libvaip-pass_vaiml_partition.so`)
- baseline reproduzível: `tools/run_whisper_encoder_xint8_probe.sh`

## Passo 2 — OGA Linux com Phi-3.5 (em progresso)

Estado atual:

- o fluxo oficial Linux da AMD para LLM ja foi confirmado pela doc
- o alvo oficial inicial continua sendo `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
- o hub ja tem `tools/run_oga_llm_linux.sh` para stage, patch e execucao do `model_benchmark`
- a tree local `runtime/llm_linux/run_phi35` existe, mas hoje os binarios dela estao com `0` bytes
- o runner do hub agora detecta placeholders vazios e falha cedo
- ainda falta materializar uma tree Linux real via `RYZEN_AI_INSTALLATION_PATH` e os blobs reais do modelo `Phi-3.5`

Objetivo:

- ter `model_benchmark` rodando de verdade no Linux com o modelo oficial `Phi-3.5`
- validar que a trilha `OGA` Linux do host deixou de ser placeholder
- usar esse baseline para abrir a frente `Qwen3` com o menor risco possivel

Desafio técnico:

- o host ainda nao tem a tree Linux oficial materializada
- o modelo oficial ainda precisa vir com blobs reais `git-lfs`
- a tree `run_phi35` atual nao serve como prova, porque hoje ela e apenas placeholder

Critério de sucesso:

- `tools/run_oga_llm_linux.sh` executando `Phi-3.5` no Linux host
- geracao real saindo do `model_benchmark`
- runtime e modelo materializados sem placeholders vazios nem ponteiros `git-lfs`

## Passo 3 — Qwen3-14B híbrido oficial AMD

Estado atual:

- a release `Ryzen AI 1.7.0` da AMD ja cita `Qwen-3-14b-Instruct`
- o wrapper `tools/run_qwen3_14b_hybrid.sh` agora fixa `RUN_DIR` e `MODEL_NAME` para esse alvo
- o wrapper foi corrigido e hoje falha no ponto certo: falta `RYZEN_AI_VENV/RYZEN_AI_INSTALLATION_PATH` com a tree Linux oficial da AMD
- esse passo continua dependente do passo 2, porque usa a mesma stack `OGA` Linux

Objetivo:

- subir `Qwen3-14B-onnx-ryzenai-1.7-hybrid` no caminho hibrido oficial da AMD
- medir se o host aguenta a trilha oficial com `96 GB` de VRAM reservados para a iGPU

Critério de sucesso:

- o wrapper `tools/run_qwen3_14b_hybrid.sh` executa o `model_benchmark`
- existe geracao real com `Qwen3-14B` no host Linux

## Passo 4 — Qwen3 grande em GPU

Estado atual:

- o kernel ja expoe `/dev/kfd` e `/dev/dri/renderD128`
- o kernel `amdgpu` ja confirma `98304M` de VRAM
- o host ainda nao tem `rocminfo`, `rocm-smi`, `hipcc`, `torch`, `transformers`, `accelerate` nem `vllm`
- mas a trilha em container `ROCm/PyTorch` ja esta funcional
- `Qwen3-4B` roda limpo na iGPU
- no split antigo `96 GB iGPU / 32 GB CPU`, `Qwen3-14B` entrava em swap pesada e `Qwen3-32B` morria por `OOM`
- no split atual `64 GB iGPU / 64 GB CPU`, `Qwen3-14B` e `Qwen3-32B` passaram a rodar de verdade na iGPU
- `tools/check_qwen3_gpu_env.sh` segue útil como preflight do host, e `tools/run_qwen3_gpu_measured.sh` agora fecha a medição reproduzível do container

Objetivo:

- manter `64/64` como split de referência para `Qwen3` grande em GPU-only
- se insistir em `OGA hybrid`, voltar o foco para materializar a tree Linux oficial da AMD

Leitura correta:

- nesta familia `Qwen3`, o proximo alvo natural acima de `14B` para GPU-only tende a ser `32B`, nao `72B`
- esta frente e separada da trilha oficial AMD `OGA hybrid`
- o split atual `64/64` resolveu o gargalo que existia no `96/32`

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
