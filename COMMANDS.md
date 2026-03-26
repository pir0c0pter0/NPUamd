# Comandos Úteis

## Verificação do kernel

```bash
lspci -nnk | rg -i 'neural|npu|amd'
```

Para distinguir `PHX/HPT` de `STX/KRK` rapidamente:

```bash
lspci -nn | rg '1022:17f0|1022:1502'
```

Resultado esperado atual:

- existe a NPU `1022:17f0`
- o host atual é `STX/KRK`, rev `11`

```bash
lsmod | rg amdxdna
```

```bash
ls -l /dev/accel /sys/class/accel
```

## Diretórios principais

```bash
ls -lah /var/home/mariostjr/amd-rai-linux/ryzen14
```

```bash
ls -lah /var/home/mariostjr/xdna-driver
```

```bash
ls -lah /var/home/mariostjr/xrt-ve2
```

## Teste do XRT local

```bash
source /var/home/mariostjr/xrt-ve2/setup.sh >/dev/null
/var/home/mariostjr/xrt-ve2/bin/xrt-smi examine
```

Resultado esperado atual:

- ainda retorna `0 devices found`

## Teste do provider AMD em container

Base usada:

```bash
podman run --rm --security-opt label=disable \
  --device /dev/accel/accel0 \
  -v /sys/class/accel:/sys/class/accel:ro \
  -v /var/home/mariostjr/amd-rai-linux:/work/amd-rai-linux:ro \
  docker.io/library/ubuntu:22.04 bash
```

Pacotes úteis no container:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-pip \
  libboost-filesystem1.74.0 \
  libboost-program-options1.74.0 \
  libboost-system1.74.0
```

Instalação Python:

```bash
python3 -m pip install --no-cache-dir onnxruntime==1.20.1 numpy
```

Carga manual das bibliotecas:

```bash
LD_LIBRARY_PATH=/work/amd-rai-linux/ryzen14 python3 -c "
import ctypes, os
base='/work/amd-rai-linux/ryzen14'
ctypes.CDLL(os.path.join(base,'libonnxruntime.so.1.20.1'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(base,'libonnxruntime_providers_shared.so'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(base,'libonnxruntime_vitisai_ep.so.1.0.0'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(base,'libonnxruntime_providers_vitisai.so'), mode=ctypes.RTLD_GLOBAL)
print('manual load ok')
"
```

## Probe nativo atual do VitisAI

Header oficial salvo no hub:

```bash
ls -lah /var/home/mariostjr/Documents/hubs/NPUamd/third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h
```

Execução reproduzível no container com o hub montado:

```bash
podman run -d --name vitis-ubuntu22-hub --security-opt label=disable \
  -v /var/home/mariostjr/amd-rai-linux:/work/amd-rai-linux:ro \
  -v /var/home/mariostjr/Documents/hubs/NPUamd:/work/hub:ro \
  docker.io/library/ubuntu:22.04 sleep infinity
```

```bash
podman exec vitis-ubuntu22-hub bash -lc '
set -euo pipefail
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-pip build-essential \
  libboost-filesystem1.74.0 \
  libboost-program-options1.74.0 \
  libboost-system1.74.0 >/dev/null
python3 -m pip install --no-cache-dir onnx >/dev/null
python3 /work/hub/tools/make_minimal_identity_model.py /tmp/minimal_identity_ir10.onnx >/dev/null
gcc -I/work/hub/third_party/onnxruntime/include \
  -L/work/amd-rai-linux/ryzen14 \
  -Wl,-rpath,/work/amd-rai-linux/ryzen14 \
  -o /tmp/probe_vitisai \
  /work/hub/tools/probe_vitisai.c \
  -lonnxruntime
export LD_LIBRARY_PATH=/work/amd-rai-linux/ryzen14
/tmp/probe_vitisai /tmp/minimal_identity_ir10.onnx
'
```

Resultado esperado atual:

- a sessão é criada com sucesso
- a inferência mínima retorna `Y=42` para `X=42`
- o modelo `Identity` continua inteiro em `CPUExecutionProvider`

## Teste nativo no host Linux

Script reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_native.sh
```

O que ele faz:

- materializa um shim local com `libpython3.10.so.1.0` e `Boost 1.74`
- reaproveita o `minimal_identity_ir10.onnx` já presente no overlay local
- compila o probe C no host
- executa a sessão nativamente no Linux host

Para passar um modelo explícito:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_native.sh /caminho/para/modelo.onnx
```

Resultado esperado atual:

- a sessão é criada com sucesso no host
- o probe detecta nome, tipo e shape reais do input/output
- o provider sobe, mas o modelo atual continua inteiro em `CPUExecutionProvider`

## Probe de partição com CNN pequeno

Runner reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_partition_probe_native.sh
```

O que ele faz:

- gera um CNN pequeno em `Ubuntu 22.04` via `tools/make_probe_cnn_model.py`
- copia o `.onnx` para o hub
- compila o probe C no host
- executa o teste nativo com `config_file`, `cache_dir` e `cache_key`

Resultado esperado atual:

- a sessão sobe e infere com input/output reais
- o log mostra `Vitis AI EP`
- o melhor resultado atual ainda é:
  ` [Vitis AI EP] No. of Operators :   CPU     8 `
- isto ainda nao prova offload real

Para usar um caminho de modelo explícito:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_partition_probe_native.sh \
  /var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/models/probe_small_cnn_opset17_explicit_kernel.onnx
```

## Teste BF16 com config mínimo do exemplo AMD

Config salvo no hub:

```bash
cat /var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/configs/vitisai_probe_bf16.json
```

Execução reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_native.sh \
  /var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/models/probe_small_cnn_opset17_explicit_kernel.onnx \
  config_file=/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/configs/vitisai_probe_bf16.json \
  cache_dir=/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/cache \
  cache_key=probe_small_cnn_bf16_config \
  enable_cache_file_io_in_mem=0
```

Resultado esperado atual:

- o provider sobe
- aparece fatal sobre plugin ausente:
  `libvaip-pass_vaiml_partition.so`
- o grafo continua inteiro em CPU

## Prova real de offload com ResNet18 XINT8 via Quark

Runner reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_resnet18_xint8_quark_probe.sh
```

O que ele faz:

- usa o container `vitis-ubuntu22-hub`
- garante `torch`, `torchvision`, `onnxruntime` e `amd-quark==0.10` dentro do container
- exporta `ResNet18` para `runtime/ubuntu22/models/resnet18_fp32_opset13.onnx`
- quantiza o modelo com `Quark XINT8` para `runtime/ubuntu22/models/resnet18_xint8_quark.onnx`
- executa o probe nativo no host com `cache_dir` e `cache_key` próprios

Resultado esperado atual:

- o cache gera `runtime/ubuntu22/cache/resnet18_xint8_quark/compiled.AMD_AIE2_Nx4_Overlay.xmodel`
- `runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json` mostra `CPU 2` e `NPU 164`
- o mesmo relatório marca `Actually running on NPU` com contagem `1`

Inspeção rápida do relatório:

```bash
python3 - <<'PY'
import json
path = "/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json"
with open(path) as f:
    data = json.load(f)
print(data["deviceStat"])
print(data["subgraphStat"])
PY
```

Leitura correta:

- o log do ORT ainda pode mostrar `All nodes placed on [CPUExecutionProvider]`
- isso nao invalida o offload real desta trilha
- a fonte de verdade aqui é `vitisai_ep_report.json`, nao apenas a linha de placement do ORT

## Prova real de offload com Whisper encoder XINT8

Runner reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_encoder_xint8_probe.sh
```

O que ele faz:

- reutiliza o mesmo caminho `Quark XINT8 + vaip_config_npu_2_3.json` que funcionou para `ResNet18`
- compila o probe nativo se necessario
- executa o encoder `tiny_en_encoder_xint8.onnx` no host Linux
- gera cache e relatorio do EP em `runtime/whisper/cache/tiny_en_encoder_xint8`

Resultado esperado atual:

- `runtime/whisper/cache/tiny_en_encoder_xint8/whisper_tiny_en_encoder_xint8/vitisai_ep_report.json` mostra `CPU 122` e `NPU 416`
- o mesmo relatorio marca `Actually running on NPU` com contagem `21`

Leitura correta desses numeros:

- `NPU 416 / CPU 122` e particionamento de operadores, nao tempo
- isso significa que `416` ops foram para a NPU e `122` ficaram na CPU
- isso prova offload real, mas nao mede speedup por si so

Inspeção rápida do relatório:

```bash
python3 - <<'PY'
import json
path = "/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/cache/tiny_en_encoder_xint8/whisper_tiny_en_encoder_xint8/vitisai_ep_report.json"
with open(path) as f:
    data = json.load(f)
print(data["deviceStat"])
print(data["subgraphStat"])
PY
```

## Estado atual do Whisper decoder XINT8

Runner reproduzível atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_decoder_xint8_probe.sh
```

O que ele faz:

- carrega `tiny_en_decoder_xint8.onnx`
- roda o decoder com `2` inputs (`x` e `xa`) via `onnxruntime`
- gera cache e relatorio do EP em `runtime/whisper/cache/xint8_decoder`

Resultado esperado atual:

- a sessao sobe e a inferencia termina com sucesso
- `runtime/whisper/cache/xint8_decoder/whisper_dec_xint8/vitisai_ep_report.json` mostra `CPU 934`
- o relatorio nao traz `subgraphStat`

Leitura correta:

- o decoder quantizado ja esta operacional no host
- isso ainda nao prova offload real para NPU
- o gap imediato do hub e fazer esse relatorio sair de `CPU only`
- `CPU 934` significa zero offload: todos os `934` ops ficaram na CPU

Inspeção rápida do relatório:

```bash
python3 - <<'PY'
import json
path = "/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/cache/xint8_decoder/whisper_dec_xint8/vitisai_ep_report.json"
with open(path) as f:
    data = json.load(f)
print(data["deviceStat"])
print(data.get("subgraphStat"))
PY
```

## Pipeline de transcrição Whisper

## Latência observada do Whisper encoder XINT8

Medição operacional registrada em `2026-03-26` para `runtime/whisper/sample_hf_1.flac`:

- CPU warm, com sessao reutilizada: cerca de `0.061 s`
- NPU warm, no runner atual do hub com cache quente: cerca de `0.26 s` a `0.32 s`
- NPU cold na primeira execucao: cerca de `7.75 s`

Leitura correta:

- no estado atual do hub, este encoder pequeno ainda nao ganha da CPU em latencia
- o valor principal da trilha NPU hoje e prova de offload real
- compile/cache da primeira execucao pesa bastante no caminho NPU

Runner validado atual do hub:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_full_hybrid.sh \
  --audio /var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/sample_hf_1.flac
```

Leitura correta:

- o wrapper sobe o container `vitis-ubuntu22-hub`
- ele compila `tools/whisper_encode_dump.c`
- ele roda o encoder XINT8 na NPU e o decoder FP32 em CPU
- hoje o resultado validado para `sample_hf_1.flac` ainda e parcial: `"I'm"`
- isso prova a trilha ponta a ponta com audio real, mas ainda nao prova transcricao correta

Detalhe importante do prompt:

- para `openai/whisper-tiny.en`, o prefixo correto desta trilha e `tokenizer.prefix_tokens`
- na pratica, isso significa `[50257, 50362]`
- o prompt longo `[50257, 50258, 50358, 50362]` fazia o caminho quantizado colapsar em `EOT` imediato

## Scaffold antigo no host

Runner antigo do hub:

```bash
python3 /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_npu_transcribe.py --test --device npu -v
```

Para usar um `.wav` real:

```bash
python3 /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_npu_transcribe.py --audio /caminho/para/audio.wav --device npu -v
```

Leitura correta:

- o script ja costura preprocessamento, encoder e decoder
- ele agora usa o mesmo prefixo compacto do modelo `.en`
- hoje ele continua sendo scaffold util, nao a trilha validada de ponta a ponta
- hoje ele ainda nao vale como prova final de NPU para Whisper completo, porque o decoder continua em CPU

## Requantização atual do Whisper encoder

Runner reproduzível atual dentro do container:

```bash
podman exec vitis-ubuntu22-hub bash -lc '
python3 /work/hub/tools/quantize_whisper_encoder_xint8.py \
  --samples 32 \
  --audio /work/hub/runtime/whisper/whisper_hello.wav \
  --audio /work/hub/runtime/whisper/sample_hf_1.flac \
  --output /tmp/tiny_en_encoder_xint8_calib.onnx
'
```

Para copiar o modelo de volta ao hub:

```bash
podman cp vitis-ubuntu22-hub:/tmp/tiny_en_encoder_xint8_calib.onnx \
  /var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_encoder_xint8.onnx
```

Leitura correta:

- a quantizacao atual do encoder nao usa mais ruído aleatório como calibração
- ela usa `WhisperFeatureExtractor` em audio real do hub
- isso melhorou o pipeline hibrido de `EOT` imediato para uma saida parcial nao vazia

## Comparação útil: QDQ puro do ORT vs Quark XINT8

Para reproduzir o caso que ainda falha:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_native.sh \
  /var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/models/resnet18_xint8_qdq.onnx \
  cache_dir=/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/cache \
  cache_key=resnet18_xint8_qdq \
  enable_cache_file_io_in_mem=0
```

Resultado esperado atual:

- o `QDQ` padrão do ORT continua em `CPU 199`
- isso mostra que nem todo `QDQ` genérico serve como prova de NPU para a stack AMD

## Teste de Qwen simples no host Linux

Instalação local do runtime JS:

```bash
cd /var/home/mariostjr/Documents/hubs/NPUamd
npm install
```

Execução reproduzível atual:

```bash
cd /var/home/mariostjr/Documents/hubs/NPUamd
npm run test:qwen
```

Arquivos envolvidos:

- `package.json`
- `tools/test_qwen_onnx.mjs`
- cache em `.cache/huggingface`

Resultado esperado atual:

- o modelo `onnx-community/Qwen2.5-0.5B-Instruct` carrega no host
- a geração retorna texto curto com sucesso
- o teste roda em CPU via `@huggingface/transformers`
- este teste nao prova uso de NPU

## Observações de erro já vistas

Erros importantes já encontrados:

```text
0 devices found
```

```text
undefined symbol: Provider_GetHost
```

```text
file too short
```

```text
cannot open shared object file: No such file or directory
```

```text
cannot open shared object file: Permission denied
```

```text
cannot open plugin: name_ vaip-pass_vaiml_partition so_name_ libvaip-pass_vaiml_partition.so
```

```text
[Vitis AI EP] No. of Operators :   CPU     8
```

```text
Attr kernel has type REQUIRED, but not set
```

## Regra prática

Quando uma `.so` vier muito pequena ou com conteúdo estranho:

- verificar se ela é alias/symlink
- buscar a versão real `.so.x.y.z`
- corrigir o symlink local

Quando um `.pt` do clone local vier muito pequeno:

- verificar se ele é ponteiro `git-lfs`
- nao tratar o ponteiro como checkpoint real
- confirmar o `size` do blob no próprio arquivo texto

## Verificação de placeholder `git-lfs`

Para inspecionar o checkpoint oficial de ResNet baixado só como ponteiro:

```bash
sed -n '1,20p' /var/home/mariostjr/RyzenAI-SW/CNN-examples/getting_started_resnet/bf16/models/resnet_trained_for_cifar10.pt
```

Resultado esperado atual:

- o arquivo começa com `version https://git-lfs.github.com/spec/v1`
- o blob real indicado ali tem `94908320` bytes

## Inventário local de artefatos `OGA`

Para confirmar que o runtime `OGA` ainda nao está materializado localmente:

```bash
find /var/home/mariostjr -maxdepth 6 \
  \( -type d -path '*/venv/deployment' -o -type f -iname 'model_benchmark' -o -type f -iname 'libonnx_custom_ops.so' -o -type f -iname 'libryzen_mm.so' \) \
  | sort
```

Resultado esperado atual:

- nao retorna a tree Linux recente esperada pela doc oficial `llm_linux`

## Fluxo oficial `LLM on Linux`

Runner atual do hub para a trilha oficial `Phi-3.5`:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_oga_llm_linux.sh \
  --prepare-only \
  --model-dir /caminho/para/Phi-3.5-mini-instruct-onnx-ryzenai-npu
```

O que ele faz:

- procura a tree Linux da AMD via `RYZEN_AI_VENV` ou `RYZEN_AI_INSTALLATION_PATH`
- copia `deployment/`, `model_benchmark` e `amd_genai_prompt.txt` para `runtime/llm_linux/run_phi35`
- aplica o patch Linux oficial em `genai_config.json`
- normaliza barras em `.cache/MatMulNBits_2_0_meta.json`
- falha cedo se a tree Linux da AMD, os blobs reais do modelo ou os binarios staged estiverem vazios

Para rodar o benchmark quando a tree oficial e o modelo completo ja estiverem materializados:

```bash
export RYZEN_AI_INSTALLATION_PATH=/caminho/para/ryzen_ai-1.6.1/venv
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_oga_llm_linux.sh \
  --model-dir /caminho/para/Phi-3.5-mini-instruct-onnx-ryzenai-npu
```

Resultado esperado quando o ambiente estiver completo:

- `model_benchmark` sobe com `LD_LIBRARY_PATH=deployment/lib`
- o modelo `Phi-3.5` ja entra com patch Linux aplicado
- aparece geracao real e metricas de prompt/token na linha oficial da AMD

## Wrapper do `Qwen3-14B` híbrido oficial AMD

Runner atual do hub:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_qwen3_14b_hybrid.sh \
  --prepare-only
```

O que ele faz:

- fixa `RUN_DIR=runtime/llm_linux/run_qwen3_14b_hybrid`
- fixa `MODEL_NAME=Qwen3-14B-onnx-ryzenai-1.7-hybrid`
- reaproveita o mesmo stage/patch do runner `OGA` generico

Para apontar para o modelo materializado:

```bash
export MODEL_DIR=/caminho/para/Qwen3-14B-onnx-ryzenai-1.7-hybrid
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_qwen3_14b_hybrid.sh
```

Leitura correta:

- este e o alvo oficial AMD para o passo seguinte depois de `Phi-3.5`
- ele depende da mesma tree Linux `RYZEN_AI_INSTALLATION_PATH`
- no host atual, o erro correto agora e:
  `missing RYZEN_AI_VENV/RYZEN_AI_INSTALLATION_PATH`

## Preflight do `Qwen3` grande em GPU

Checker atual do hub:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/check_qwen3_gpu_env.sh
```

O que ele checa:

- `MemTotal` visivel ao CPU
- VRAM exposta pelo kernel `amdgpu`
- presenca de `/dev/kfd` e `/dev/dri/renderD128`
- presenca de `torch`, `transformers`, `accelerate`
- presenca de `rocminfo`

Resultado esperado hoje:

- o kernel ja reporta `98304M` de VRAM na iGPU
- os device nodes existem
- o checker do host ainda falha por falta de stack `ROCm/PyTorch` no user-space local
- isso nao invalida a trilha em container, que ja foi validada separadamente

## Runner do `Qwen3` em GPU via container ROCm

Runner simples atual:

```bash
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_qwen3_gpu_container.sh
```

O que ele faz:

- garante o container `qwen3-gpu-pytorch`
- instala `transformers`, `accelerate`, `sentencepiece` e `safetensors`
- roda `tools/run_qwen3_gpu_transformers.py` dentro de `docker.io/rocm/pytorch:latest`

Resultado esperado hoje:

- `Qwen/Qwen3-4B` carrega e gera de verdade na iGPU

## Runner medido do `Qwen3` em GPU

Runner reproduzível atual:

```bash
MODEL_ID='Qwen/Qwen3-4B' MAX_NEW_TOKENS=32 \
bash /var/home/mariostjr/Documents/hubs/NPUamd/tools/run_qwen3_gpu_measured.sh
```

O que ele faz:

- chama o runner ROCm em container
- resolve o PID real do `python3` no host
- coleta CSV com `cpu_pct`, `rss_mib`, `gpu_busy_pct`, `gpu_vram_used_gib`, `gpu_power_w` e estado da NPU
- grava log e CSV em `runtime/gpu_runs/`

Leitura correta dos testes atuais:

- `Qwen3-4B` fecha limpo: cerca de `9.38 GiB` de VRAM, pico de `87%` de `gpu_busy_percent`, cerca de `37 W`
- no split antigo `96/32`, `Qwen3-14B` entra em swap pesada e `Qwen3-32B` morre por `OOM`
- no split atual `64/64`, `Qwen3-14B` fecha com cerca de `28.96 GiB` de VRAM e `Qwen3-32B` fecha com cerca de `58.03 GiB` de VRAM

## Verificação de placeholder `git-lfs` do `Phi-3.5`

Para confirmar se o clone do modelo ainda esta em ponteiros `git-lfs`:

```bash
sed -n '1,20p' /caminho/para/Phi-3.5-mini-instruct-onnx-ryzenai-npu/fusion.onnx
```

Resultado esperado quando ainda nao materializou:

- o arquivo começa com `version https://git-lfs.github.com/spec/v1`
- o blob real ainda nao esta no disco

## Verificação do estado do container de exportação

Para saber se `torch` e `torchvision` ficaram instalados no `vitis-ubuntu22-hub` depois da interrupção:

```bash
podman exec vitis-ubuntu22-hub bash -lc 'python3 -m pip show torch torchvision'
```

Leitura correta:

- se ambos aparecerem, o container já pode ser reaproveitado para exportação ONNX
- se nao aparecerem, a instalação precisa ser retomada ou refeita

## Ordem de retomada

Ao retomar depois de limpar a conversa, seguir esta ordem:

1. manter `tools/run_resnet18_xint8_quark_probe.sh` e `tools/run_whisper_encoder_xint8_probe.sh` como baselines que nao podem regredir
2. fechar `tools/run_oga_llm_linux.sh` com `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
3. subir `tools/run_qwen3_14b_hybrid.sh`
4. abrir a frente `Qwen3` grande em GPU so depois do preflight `tools/check_qwen3_gpu_env.sh`
