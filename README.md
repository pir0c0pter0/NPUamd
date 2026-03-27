# AMD NPU on Linux: real status on STX/KRK

Este repositorio documenta, de forma pratica e reproduzivel, o estado real de bring-up da NPU AMD em Linux no host abaixo:

- plataforma `STX/KRK`
- PCI ID da NPU `1022:17f0`
- driver `amdxdna`
- foco em `ONNX Runtime + VitisAIExecutionProvider`, `Whisper` e `LLM local`

O objetivo aqui nao e repetir marketing nem baseline de CPU. O objetivo e responder:

1. a NPU aparece no Linux?
2. a stack AMD sobe de verdade?
3. existe offload real para NPU?
4. qual o caminho atual para fechar `OGA` em Linux com `Phi-3.5`?

## Resposta curta

Hoje a resposta e:

- sim, a NPU existe e o kernel a reconhece
- sim, a stack `VitisAIExecutionProvider` sobe no Linux host e em `Ubuntu 22.04`
- sim, ja existe prova forte e reproduzivel de offload real para a NPU no host Linux com `ResNet18 XINT8`
- sim, ja existe prova forte e reproduzivel de offload de transformer para a NPU com `Whisper tiny.en encoder XINT8`
- sim, o decoder Whisper FP32 foi compilado via VAIML com 99.6% dos ops na NPU (235/236), mas a inferencia crasha por incompatibilidade ABI entre VAIML 1.4.0 e XRT 2.23.0
- sim, ja existe pipeline hibrido ponta a ponta com audio real usando encoder na NPU e decoder FP32 em CPU; hoje o melhor resultado validado ainda e parcial (`"I'm"` no `sample_hf_1.flac`)
- nao, `LLM` ainda nao esta fechado com prova final de NPU neste host; `FastFlowLM` e o caminho mais viavel (suporta `Qwen3.5:9B` na NPU via Linux), mas precisa de kernel 7.0+ e firmware NPU >= 1.1.0.0
- em paralelo, a trilha `Qwen3` grande em GPU ja esta validada em container `ROCm/PyTorch`: com o split atual `64 GB iGPU / 64 GB CPU`, `Qwen3-14B` e `Qwen3-32B` carregam e geram de verdade na iGPU

## O que ja foi provado

### 1. Hardware e kernel

Ja foi validado:

- `lspci` mostra a NPU AMD
- o modulo `amdxdna` esta carregado
- existe device de aceleracao exposto pelo kernel

Leitura correta:

- o problema principal deixou de ser "hardware nao detectado"
- o ponto restante esta em user-space e runtime

### 2. Provider AMD subindo no Linux

Ja foi validado:

- `VitisAIExecutionProvider` sobe por C API nativa
- o probe funciona no host Linux
- o probe tambem funciona em container `Ubuntu 22.04`
- o caminho nativo evita o `Python 3.14` do host e reutiliza userspace compativel `Python 3.10 + Boost 1.74`

### 3. Primeira prova real de offload para NPU

O primeiro caso que realmente fez offload foi:

- `runtime/ubuntu22/models/resnet18_xint8_quark.onnx`

Trilha usada:

- exportacao de `ResNet18`
- quantizacao com `amd-quark==0.10`
- execucao via `VitisAIExecutionProvider`

Prova forte:

- `runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json` mostra `CPU 2` e `NPU 164`
- o mesmo relatorio marca `Actually running on NPU 1`

Nuance importante:

- o log do ORT ainda pode dizer algo ambiguo sobre placement em CPU
- a fonte de verdade aqui e o relatorio do EP, nao uma linha isolada do log

### 4. Primeira prova real de offload de transformer

O segundo caso que realmente fez offload foi:

- `runtime/whisper/models/tiny_en_encoder_xint8.onnx`

Trilha usada:

- download do encoder `amd/whisper-tiny-en-onnx-npu`
- quantizacao com `amd-quark==0.10`
- execucao via `VitisAIExecutionProvider` com `vaip_config_npu_2_3.json`

Prova forte:

- `runtime/whisper/cache/tiny_en_encoder_xint8/whisper_tiny_en_encoder_xint8/vitisai_ep_report.json` mostra `CPU 122` e `NPU 416`
- o mesmo relatorio marca `Actually running on NPU 21`

Leitura correta:

- o primeiro transformer com offload real neste host ja existe
- o caminho que funciona nesta tree Linux e `Quark XINT8 + DPU/DD`
- o caminho VAIML/BF16 continua bloqueado pela ausencia de `libvaip-pass_vaiml_partition.so`
- `NPU 416 / CPU 122` e contagem de operadores particionados por device, nao medicao de tempo
- na medicao operacional atual do hub, esse encoder pequeno ainda nao ganha da CPU em latencia; o valor desta trilha hoje e prova de offload real, nao speedup

### 5. O que ainda nao vale como prova

Ainda nao vale como prova de NPU:

- baseline de `Qwen` em `transformers.js` no host, porque hoje ele roda em CPU
- `tiny_en_decoder_xint8.onnx`, porque hoje o relatorio ainda mostra `CPU 934` e nao gera `subgraphStat`
- CNN sintetico pequeno com `2 Conv`, porque continua inteiro em CPU
- `xrt-smi` atual, porque ainda retorna `0 devices found`
- checkpoint pequeno `.pt` do exemplo AMD, porque hoje ele e so ponteiro `git-lfs`

## Estado atual de LLM no Linux

O ponto mais importante desta rodada:

- a documentacao oficial atual da AMD ja tem uma pagina explicita de `Running LLM on Linux`
- essa pagina usa como referencia o modelo `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`

Isso muda a leitura do projeto:

- o bloqueio nao e mais "talvez Linux nao tenha fluxo oficial"
- o bloqueio agora e materializar a instalacao Linux recente da AMD e os blobs reais do modelo

No host atual ainda faltam:

- uma tree `RYZEN_AI_INSTALLATION_PATH` do tipo `ryzen_ai-1.6.1+/venv`
- `deployment/lib/libonnx_custom_ops.so`
- `deployment/lib/libryzen_mm.so`
- `model_benchmark`
- blobs reais `git-lfs` do modelo `Phi-3.5`

Estado local importante desta rodada:

- existe uma tree `runtime/llm_linux/run_phi35`
- mas hoje `libonnxruntime-genai.so`, `libonnx_custom_ops.so`, `libryzen_mm.so`, `model_benchmark` e `amd_genai_prompt.txt` nela estao com `0` bytes
- isso nao vale como runtime materializado
- o runner do hub agora falha cedo nesse caso e exige restage a partir da tree Linux oficial

Alternativa em andamento:

- `FastFlowLM` 0.9.37 como runtime NPU-first no Linux
- suporta `Qwen3.5:9B`, `Qwen3:8B`, `Whisper` e outros modelos direto na NPU
- requer kernel 7.0+ com driver `amdxdna` nativo e firmware NPU >= 1.1.0.0
- Bazzite/ostree bloqueia troca de kernel por conflito com gaming kmods; proximo passo e instalar distro com kernel 7 nativo (Arch ou Fedora 44)
- repo clonado em `~/FastFlowLM` com submodules prontos pra build

## Estado atual do Qwen3 em GPU

Tambem ja existe uma trilha reproduzivel de `Qwen3` em GPU-only no host, mas ela depende de container `ROCm/PyTorch`, nao da stack oficial `OGA hybrid` da AMD.

Estado validado agora:

- o container `qwen3-gpu-pytorch` com `docker.io/rocm/pytorch:latest` sobe com acesso real a `/dev/kfd` e `/dev/dri`
- `Qwen/Qwen3-4B` carrega e gera de verdade na iGPU
- no teste medido atual do `Qwen3-4B`, o modelo carregou em cerca de `2.50 s`, gerou em cerca de `2.84 s`, bateu pico de `9.38 GiB` de VRAM, `87%` de `gpu_busy_percent` e cerca de `37 W`
- no split antigo `96/32`, `Qwen/Qwen3-32B` morria por `OOM` e `Qwen/Qwen3-14B` entrava em swap pesada
- no split atual `64/64`, `Qwen/Qwen3-14B` carrega em cerca de `12.10 s`, gera em cerca de `7.55 s`, bate pico de `28.96 GiB` de VRAM, `99%` de `gpu_busy_percent` e cerca de `34 W`
- no split atual `64/64`, `Qwen/Qwen3-32B` carrega em cerca de `85.54 s`, gera em cerca de `13.12 s`, bate pico de `58.03 GiB` de VRAM, `98%` de `gpu_busy_percent` e cerca de `37 W`

Leitura correta:

- o gargalo real do `Qwen3` grande neste host nao era a VRAM
- o gargalo real era a RAM visivel do lado CPU quando a BIOS estava em `96/32`
- com `64/64`, `4B`, `14B` e `32B` ficam viaveis na trilha GPU-only em container

## Estrutura deste repositorio

Arquivos principais:

- `README.md`: visao publica de alto nivel
- `STATUS.md`: estado tecnico detalhado
- `NEXT-STEPS.md`: prioridade objetiva dos proximos passos
- `COMMANDS.md`: base operacional de reproducao
- `AGENTS.md`: instrucoes de retomada do hub

Scripts principais:

- `tools/run_vitisai_probe_native.sh`
- `tools/run_vitisai_partition_probe_native.sh`
- `tools/run_resnet18_xint8_quark_probe.sh`
- `tools/run_whisper_encoder_xint8_probe.sh`
- `tools/run_whisper_decoder_xint8_probe.sh`
- `tools/run_whisper_full_hybrid.sh`
- `tools/run_whisper_hybrid_transcribe.py`
- `tools/run_whisper_npu_transcribe.py`
- `tools/whisper_encode_dump.c`
- `tools/run_oga_llm_linux.sh`
- `tools/patch_oga_linux_model.py`
- `tools/run_qwen3_14b_hybrid.sh`
- `tools/run_qwen3_gpu_container.sh`
- `tools/run_qwen3_gpu_measured.sh`
- `tools/run_qwen3_gpu_transformers.py`
- `tools/monitor_apu_usage.sh`

## Melhor baseline hoje

Se eu precisasse retomar do zero com o menor risco tecnico, eu faria nesta ordem:

1. manter `resnet18_xint8_quark.onnx` como baseline reproduzivel do primeiro offload real
2. manter `tiny_en_encoder_xint8.onnx` como baseline reproduzivel do primeiro transformer na NPU
3. retomar `OGA` em Linux usando primeiro `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
4. subir `tools/run_qwen3_14b_hybrid.sh` como alvo oficial de LLM hibrido AMD
5. so depois abrir a frente separada de `Qwen3` grande em GPU

## Repositorios AMD usados

Repositorios oficiais usados diretamente nesta investigacao:

- `amd/RyzenAI-SW`
  - <https://github.com/amd/RyzenAI-SW>
- `amd/xdna-driver`
  - <https://github.com/amd/xdna-driver>

Modelos AMD usados ou escolhidos como alvo:

- `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
  - <https://huggingface.co/amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu>
- `amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix`
  - <https://huggingface.co/amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix>

Documentacao AMD consultada de forma central:

- `Running LLM on Linux`
  - <https://ryzenai.docs.amd.com/en/latest/llm_linux.html>
- `OnnxRuntime GenAI (OGA) Flow`
  - <https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html>
- `OGA NPU Execution Mode`
  - <https://ryzenai.docs.amd.com/en/latest/npu_oga.html>
- `Model Compilation and Deployment`
  - <https://ryzenai.docs.amd.com/en/latest/modelrun.html>

## O que este repositorio nao faz

Este repositorio nao assume que:

- qualquer baseline de CPU prova uso da NPU
- `Ollama`, `GGUF`, `transformers.js` ou `Qwen` generico usem a NPU da AMD automaticamente
- `xrt-smi` atual seja prova final de sucesso
- o fluxo correto passe por Windows

Tambem nao trata como caminho principal:

- voltar para Windows
- insistir em `Python 3.14` do host
- rebaixar o problema para "hardware nao detectado"

## Como reproduzir o estado atual

Para o baseline de offload real:

```bash
bash tools/run_resnet18_xint8_quark_probe.sh
```

Para o baseline de transformer na NPU:

```bash
bash tools/run_whisper_encoder_xint8_probe.sh
```

Para ver o estado atual do decoder Whisper:

```bash
bash tools/run_whisper_decoder_xint8_probe.sh
```

Leitura correta desse ultimo comando:

- hoje ele sobe e infere o decoder XINT8 no host
- hoje o relatorio ainda sai `CPU 934`
- isso ainda nao prova offload do decoder

Para exercitar o pipeline completo de transcricao:

```bash
bash tools/run_whisper_full_hybrid.sh --audio /var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/sample_hf_1.flac
```

Leitura correta desse comando:

- ele ja costura preprocessamento, encoder e decoder em audio real
- o encoder continua saindo com `NPU 416 / CPU 122`
- o prompt correto para `whisper-tiny.en` nesta trilha e o prefixo compacto `[50257, 50362]`
- hoje ele valida o caminho hibrido `encoder na NPU + decoder FP32 em CPU`
- hoje o melhor resultado validado no `sample_hf_1.flac` ainda e parcial (`"I'm"`), entao isso ainda nao vale como prova final de Whisper completo

Para a trilha antiga em Python no host:

```bash
python3 tools/run_whisper_npu_transcribe.py --test --device npu -v
```

Leitura correta desse ultimo comando:

- ele segue existindo como scaffold de alto nivel
- ele agora usa o mesmo prefixo compacto do modelo `.en`
- a trilha validada de ponta a ponta hoje continua sendo o wrapper em container

Para o probe nativo do provider AMD:

```bash
bash tools/run_vitisai_probe_native.sh
```

Para a trilha oficial `OGA Linux` com `Phi-3.5`:

```bash
bash tools/run_oga_llm_linux.sh --prepare-only --model-dir /caminho/para/Phi-3.5-mini-instruct-onnx-ryzenai-npu
```

Leitura correta desse ultimo comando:

- ele ja prepara a trilha oficial Linux no hub
- ele falha cedo se faltar a tree `RYZEN_AI_INSTALLATION_PATH`
- ele falha cedo se o modelo ainda estiver so em ponteiros `git-lfs`

## Estado atual em uma linha

O host Linux ja provou offload real para CNN e transformer na NPU AMD; o decoder Whisper FP32 compilou 99.6% na NPU via VAIML mas crasha na inferencia por ABI mismatch XRT/VAIML; o proximo passo concreto e instalar uma distro com kernel 7.0+ (Arch ou Fedora 44) e rodar `FastFlowLM` com `Qwen3.5:9B` na NPU como primeiro LLM real no device.
