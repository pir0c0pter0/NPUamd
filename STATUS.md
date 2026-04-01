# Status Atual

## Objetivo

Fazer a NPU AMD funcionar no Linux host atual, com foco prático em:

- Whisper
- LLM local

## Host

- Usuário: `mariostjr`
- Sistema: CachyOS (Arch-based)
- Modelo de gestão: `pacman`
- Data deste registro: `2026-03-27`
- Kernel: `7.0.0-rc5` (compilado do source)
- Plataforma NPU confirmada agora: `STX/KRK`
- PCI ID da NPU: `1022:17f0`
- Revisão observada da NPU: `rev 11`
- GPU observada agora: `Strix Halo [Radeon 8050S/8060S]`, PCI ID `1002:1586`
- VRAM observada pelo kernel agora: `98304M`
- RAM visível ao lado CPU agora: `MemTotal 32470164 kB` (~`31.0 GiB`)

## O que já funciona

### Kernel e hardware

A NPU foi confirmada no kernel.

Já foi verificado:

- `lspci -nnk` mostra a NPU AMD
- o driver em uso é `amdxdna`
- o módulo `amdxdna` está carregado
- existe device de aceleração exposto pelo kernel

Conclusão:

- a NPU existe
- o kernel a inicializa
- o problema atual é user-space

### User-space AMD local

Foi criado um espelho local da stack Linux da AMD em:

- `/var/home/mariostjr/amd-rai-linux/ryzen14`

Esse diretório já contém:

- `libonnxruntime.so.1.20.1`
- `libonnxruntime_providers_shared.so`
- `libonnxruntime_providers_vitisai.so`
- `libonnxruntime_vitisai_ep.so.1.0.0`
- `libvaip-*`
- `libxir.so.3.5.0`
- `libvart-*`
- `libxcompiler-*`
- `vaip_config_npu_2_3.json`
- overlays `.xclbin`

### Ambiente compatível identificado

Foi confirmado que:

- o host `Python 3.14` nao é bom encaixe para essa stack
- `onnxruntime==1.20.1` encaixa em `Python 3.10`
- `Ubuntu 22.04` em container é um alvo melhor para validar primeiro

## O que ainda nao funciona

### XRT local

A árvore local de `xdna-driver/XRT` ainda nao enumera o device de forma útil.

Estado observado:

- `xrt-smi` executa
- retorna `0 devices found`

### ONNX Runtime + VitisAI

A stack Linux da AMD já passou da fase de carga manual de `.so`.

Estado validado agora em container `Ubuntu 22.04`:

- `libonnxruntime.so.1.20.1` local da AMD aceita `SessionOptionsAppendExecutionProvider_VitisAI`
- uma sessão ONNX é criada com sucesso por C API nativa
- uma inferência mínima `Identity` roda com sucesso
- o grafo mínimo continua inteiro em `CPUExecutionProvider`, o que é esperado para esse caso

Isso indica:

- o bloqueio já nao é hardware
- o bloqueio principal já nao é a integração final do provider
- o próximo passo é provar offload real em um modelo compatível

### Probe nativo no host Linux

Também foi validado o mesmo baseline fora do container.

Estado validado agora no host Linux:

- o probe C compila no host contra `libonnxruntime.so.1.20.1`
- o runtime sobe nativamente com `LD_LIBRARY_PATH` apontando para `/var/home/mariostjr/amd-rai-linux/ryzen14`
- as dependências `libpython3.10.so.1.0` e `Boost 1.74` podem ser materializadas localmente a partir do overlay já existente de `Ubuntu 22.04`
- a mesma inferência mínima `Identity` roda com sucesso no host
- o grafo mínimo continua inteiro em `CPUExecutionProvider`, o que segue esperado para esse caso

### Probe de partição ampliado

Nesta retomada, o probe nativo foi ampliado para sair do caso trivial `Identity`.

Estado validado agora no host Linux:

- `tools/probe_vitisai.c` agora inspeciona nome, tipo e shape reais do input/output do modelo
- foi criado `tools/make_probe_cnn_model.py` para gerar um CNN pequeno com `2 Conv`
- foi criado `tools/run_vitisai_partition_probe_native.sh` para reproduzir o teste de partição com modelo real no host
- foi criado `runtime/ubuntu22/configs/vitisai_probe_bf16.json` para testar o caminho BF16 descrito na documentação AMD
- o modelo sintético `probe_small_cnn_opset17_explicit_kernel.onnx` sobe e infere no host, mas o log continua mostrando:
  ` [Vitis AI EP] No. of Operators :   CPU     8 `
- isto significa que, mesmo fora do `Identity`, ainda nao existe partição real para NPU comprovada

Descobertas úteis deste teste:

- a primeira versão do CNN sintético disparou um fatal de XIR sobre attr `kernel` ausente; isso foi corrigido ao tornar os `Conv` mais explícitos no ONNX
- mesmo com `kernel_shape` explícito, o grafo continuou inteiro em CPU
- o config BF16 mínimo do exemplo AMD nao fecha nesta tree Linux local porque falta o plugin `libvaip-pass_vaiml_partition.so`

Conclusão prática:

- o provider sobe e tenta processar modelos reais
- o caminho atual ainda nao entrega prova de offload
- o próximo candidato deve ser um modelo oficial AMD ou um ONNX real já validado pela própria stack AMD

### Prova real de offload com ResNet18 XINT8 via Quark

Nesta retomada, o primeiro offload real da NPU foi provado no host Linux.

Estado validado agora:

- o container `vitis-ubuntu22-hub` foi validado com `torch`, `torchvision`, `amd-quark==0.10` e `onnxruntime 1.22.1`
- foi criado `tools/make_resnet18_xint8_quark_model.py` para exportar `ResNet18` e quantizar em `Quark XINT8`
- foi criado `tools/run_resnet18_xint8_quark_probe.sh` para reproduzir a trilha inteira no hub
- o runner materializa `runtime/ubuntu22/models/resnet18_fp32_opset13.onnx`
- o runner materializa `runtime/ubuntu22/models/resnet18_xint8_quark.onnx`
- o probe nativo do host gera `runtime/ubuntu22/cache/resnet18_xint8_quark/compiled.AMD_AIE2_Nx4_Overlay.xmodel`
- `runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json` mostra `CPU 2` e `NPU 164`
- o mesmo relatório registra `subgraphStat` com `NPU 1` e `Actually running on NPU 1`
- `runtime/ubuntu22/cache/resnet18_xint8_quark/context.json` registra um subgrafo grande indo de `input_QuantizeLinear_Output` até `output_QuantizeLinear_Output`

Comparação útil desta rodada:

- um `QDQ` padrão do ORT para o mesmo `ResNet18` ainda cai inteiro em CPU e gera `CPU 199`
- o `Quark XINT8` do mesmo modelo passa para `NPU 164 / CPU 2`

Nuance importante:

- o log do ORT ainda pode imprimir `All nodes placed on [CPUExecutionProvider]`
- isso nao invalida o offload real nesta trilha
- a fonte de verdade aqui é o relatório do EP `vitisai_ep_report.json`, que marca explicitamente o subgrafo `Actually running on NPU`

Conclusão prática:

- o passo 1 do hub agora foi cumprido
- nem todo `QDQ` genérico é suficiente para a stack AMD
- o dialeto de quantização do `Quark` importa para sair do `CPU only`

### Qwen simples no host Linux

Também foi validado um baseline de LLM nativo no host, separado da stack OGA da AMD.

Estado validado agora no host Linux:

- foi instalado `@huggingface/transformers` localmente no hub
- foi baixado o modelo `onnx-community/Qwen2.5-0.5B-Instruct`
- o modelo foi carregado e respondeu no host Linux
- o cache local do modelo ficou em `/var/home/mariostjr/Documents/hubs/NPUamd/.cache/huggingface`
- o teste atual usa backend CPU via `transformers.js`, sem offload para NPU

Medição simples com cache quente:

- tempo total observado: cerca de `6.11s`
- pico de memória observado: cerca de `3.1 GB`

Leitura correta deste resultado:

- isso prova apenas compatibilidade básica de LLM no host Linux
- isso nao prova uso da NPU
- isso nao deve virar trilha principal do hub

### Checkpoint oficial de CNN ainda nao materializado

Também foi verificado agora o caminho do exemplo oficial de ResNet no clone local `RyzenAI-SW`.

Estado observado:

- o arquivo `CNN-examples/getting_started_resnet/bf16/models/resnet_trained_for_cifar10.pt` existe localmente
- porém ele tem apenas `133` bytes e é texto ASCII
- o conteúdo dele é um ponteiro `git-lfs`
- o blob real apontado por ele é de `94908320` bytes
- `git lfs` nao está instalado no host neste momento

Leitura correta:

- esse checkpoint oficial ainda nao pode ser usado como base de exportação para ONNX
- antes de tentar exportar o ResNet do exemplo AMD, é preciso materializar o blob real

### Container de quantização agora está consolidado

O container `vitis-ubuntu22-hub` deixou de ser uma incógnita.

Estado correto a assumir agora:

- `torch` e `torchvision` já foram usados para exportar `ResNet18` em ONNX
- `amd-quark==0.10` já foi instalado e usado com sucesso
- `onnxruntime` dentro desse container terminou em `1.22.1`, trazido pela própria resolução de dependências do `amd-quark`
- esse container agora serve como base prática da trilha `Quark XINT8`

### OGA local ainda nao existe como runtime pronto

Também foi feito um inventário do que já existe localmente para `OGA`.

Estado observado:

- o host nao tem `RYZEN_AI_INSTALLATION_PATH` apontando para uma tree Linux recente do tipo `ryzen_ai-1.6.1+/venv`
- nao existe localmente `deployment/lib/libonnx_custom_ops.so`
- nao existe localmente `deployment/lib/libryzen_mm.so`
- nao existe localmente `model_benchmark`
- o clone local de `RyzenAI-SW` contém exemplos e documentação, mas nao materializa o runtime `OGA` em Linux por si só
- o container `vitis-ubuntu22-hub` hoje tem `onnxruntime-genai 0.9.2`, mas isso sozinho nao substitui a tree oficial `deployment/` da AMD
- a tree local `runtime/llm_linux/run_phi35` existe, mas hoje `libonnxruntime-genai.so`, `libonnx_custom_ops.so`, `libryzen_mm.so`, `model_benchmark` e `amd_genai_prompt.txt` nela estao com `0` bytes

Leitura correta:

- o passo 2 ainda nao está bloqueado por hardware
- ele está bloqueado por ausência de runtime/artefatos corretos no Linux local
- nao se deve presumir que a presença dos exemplos Python da AMD ou da wheel genérica `onnxruntime-genai` equivale à presença do runtime Linux oficial da AMD
- tambem nao se deve tratar a tree `runtime/llm_linux/run_phi35` atual como runtime pronto; hoje ela e apenas placeholder

### Fluxo oficial `LLM on Linux` agora está confirmado

Tambem foi conferida nesta retomada a documentacao oficial mais recente da AMD para LLM em Linux.

Estado confirmado agora:

- existe uma pagina oficial `Running LLM on Linux` na documentacao `Ryzen AI Software 1.7.0`
- a pagina foi atualizada em `2026-01-22`
- o exemplo de referencia usado pela propria AMD e `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
- o fluxo oficial espera uma instalacao Linux sob `RYZEN_AI_INSTALLATION_PATH`, por exemplo `<TARGET-PATH>/ryzen_ai-1.6.1/venv`
- o fluxo oficial manda copiar `deployment/`, `model_benchmark` e `amd_genai_prompt.txt` a partir dessa tree
- o fluxo oficial manda editar `genai_config.json` para usar `deployment/lib/libonnx_custom_ops.so`
- o fluxo oficial manda adicionar `hybrid_dbg_use_aie_rope=0` em `config_entries`
- o fluxo oficial manda normalizar barras em `.cache/MatMulNBits_2_0_meta.json`
- o clone cru do modelo `Phi-3.5` cai em ponteiros `git-lfs` para `fusion.onnx`, `fusion.onnx.data`, `prefill.bin` e outros blobs

### Situação oficial mais recente consultada

Foi conferida nesta retomada a documentação oficial mais recente da AMD disponível em `2026-03-24`.

Leitura consolidada:

- o hardware local `17f0` está na família suportada `STX/KRK`
- a documentacao oficial atual ja inclui uma trilha explicita de `LLM on Linux` para `Phi-3.5`
- essa trilha Linux depende de uma tree recente de instalacao com `deployment/lib/libonnx_custom_ops.so`, `libryzen_mm.so` e `model_benchmark`
- essa tree ainda nao existe neste host
- o modelo oficial `Phi-3.5` foi confirmado no Hugging Face, mas o clone cru ainda precisa materializar os blobs `git-lfs`

Implicação:

- o passo 2 deve continuar em Linux, como meta do hub
- mas ele agora tem um alvo operacional concreto
- o bloqueio deixou de ser “nao existe fluxo oficial Linux” e passou a ser “falta materializar a instalacao Linux recente e os blobs reais do modelo”

### Qwen3-14B híbrido oficial AMD

Tambem foi retomado o wrapper do `Qwen3-14B` no caminho hibrido oficial AMD.

Estado validado agora:

- o script `tools/run_oga_llm_linux.sh` tinha erros de shell e foi corrigido
- o mesmo runner deixou de assumir apenas o layout `Phi` e agora infere os artefatos obrigatorios a partir de `genai_config.json`
- isso cobre o pacote local do `Qwen3-14B hybrid`, que usa `model_jit.onnx`, `model_jit.onnx.data`, `model_jit.pb.bin`, `model_jit.bin` e `dd_plugins/`
- o helper `tools/patch_oga_linux_model.py` agora aceita a ausencia de `.cache/MatMulNBits_2_0_meta.json` quando esse metadado opcional nao vier no pacote do modelo
- o preflight do runner agora detecta tambem a falta de `deployment/lib/libonnxruntime_providers_ryzenai.so`, exigida pelo `provider_options: RyzenAI` desse pacote `Qwen3`
- o wrapper `tools/run_qwen3_14b_hybrid.sh` agora chega na validacao correta do ambiente
- o modelo `runtime/llm_linux/models/Qwen3-14B-onnx-ryzenai-1.7-hybrid` esta materializado localmente
- foi validado em `2026-03-26` que, com um runtime Linux staged nao-vazio em diretório temporario, `bash tools/run_qwen3_14b_hybrid.sh --prepare-only` fecha o stage/patch do `Qwen3` sem cair mais na suposicao antiga de layout `Phi`
- tambem foi validado em `2026-03-26` que o `onnxruntime-genai==0.12.2` generico nao basta: com `custom_ops_library` presente o load falha por falta de `deployment/lib/libonnx_custom_ops.so`; sem esse campo, o load seguinte falha por falta de `libonnxruntime_providers_ryzenai.so`
- `bash tools/run_qwen3_14b_hybrid.sh` hoje falha cedo com a mensagem correta sobre runtime Linux incompleto
- o host continua sem `RYZEN_AI_VENV` ou `RYZEN_AI_INSTALLATION_PATH`
- o host continua sem `deployment/lib/libonnxruntime-genai.so`, `libonnx_custom_ops.so`, `libryzen_mm.so`, `libonnxruntime_providers_ryzenai.so` e `model_benchmark` vindos de uma tree Linux oficial recente
- tambem foi validado em `2026-03-26` que a copia segura de `~/Downloads/ryzen_ai-1.4.0.tgz` para `~/amd-rai-linux/installers/ryzen_ai-1.4.0.tgz` preserva o original intacto, mas nao fecha este passo
- esse instalador `1.4.0` consegue popular wheels e `VOE` em container `Ubuntu 22.04`, porem nao materializa no host a tree `venv/deployment/` e `venv/LLM/examples/` esperada hoje pela doc `llm_linux`
- no `1.4.0`, o runtime de LLM fica empacotado em `npu-llm.tar.gz` e `voe/lib`, nao como tree pronta `ryzen_ai-1.6.1+/venv`
- o `npu-llm.tar.gz` do `1.4.0` traz `model_benchmark`, `run_llm`, `libonnxruntime-genai.so`, `libonnxruntime_providers_vitisai.so` e `libonnxruntime_vitis_ai_custom_ops.so`, mas nao traz `libonnx_custom_ops.so`, `libryzen_mm.so` nem `libonnxruntime_providers_ryzenai.so`
- probes limitados mostraram o corte de versao: o `onnxruntime_genai_ryzenai 0.6.0` do `1.4.0` conhece `qwen2`, mas nao `qwen3`; o `model_benchmark` retorna `Unsupported model_type in config.json: qwen3`
- forcar o pacote `Qwen3` para `model.type=qwen2` tambem nao fecha o caminho: o `model_benchmark` do `1.4.0` retorna `Unknown provider type: RyzenAI`, e a API Python `0.6.0` tambem rejeita `VitisAI`
- na outra ponta, o `onnxruntime-genai 0.12.2` generico aceita `qwen3`, mas continua exigindo `libonnxruntime_providers_ryzenai.so`; se o provider e trocado para `VitisAI`, esse build responde `VitisAI execution provider is not supported in this build`

Leitura correta:

- o bloqueio imediato do `Qwen3-14B hybrid` deixou de ser a suposicao errada de layout no hub
- o bloqueio real passou a ser duplo: falta a tree Linux oficial recente da AMD para `OGA`, e a copia `1.4.0` que ja foi inspecionada e estruturalmente antiga demais para este `Qwen3-14B-onnx-ryzenai-1.7-hybrid`
- para retomar o caminho hibrido oficial em Linux, o alvo minimo agora e uma instalacao AMD mais nova no formato `ryzen_ai-1.6.1+/venv`, alinhada com `Qwen3` e com o provider `RyzenAI`
- a copia `1.4.0` segue util para inspecao de `OGA 0.6.0 / VitisAI`, mas nao como runtime final deste modelo

### Probes avancados com tree npu-llm do 1.4.0 (2026-03-26)

Foi feita uma rodada de probes tentando usar os binarios e libs do `npu-llm` do instalador 1.4.0 para desbloquear LLM na NPU e melhorar o offload do Whisper decoder.

Descobertas sobre a tree `npu-llm`:

- `model_benchmark` (237 KB) e `run_llm` (155 KB) rodam no host com `LD_LIBRARY_PATH=npu-llm/lib`
- a tree tem 236 `.so` (~575 MB), incluindo `libonnxruntime-genai.so` (4.4 MB), `libonnxruntime_vitis_ai_custom_ops.so` (865 KB) e `libtransaction.so` (309 MB)
- a tree tem `libvaip-pass_vaiml_partition.so` **real** (3.3 MB) + 5 libs VAIML auxiliares — a lib que faltava na tree `ryzen14`
- a tree tem `vaip_llm.json` com passes `matmul_nbits` e `SSMLP`
- a tree **nao** tem `libonnx_custom_ops.so`, `libryzen_mm.so` nem `libonnxruntime_providers_ryzenai.so`

Probes de LLM com o Qwen3-14B:

- forcar `model_type=qwen2` + `provider=VitisAI` no genai_config fez o `model_benchmark` do 1.4.0 aceitar o modelo e carregar o VitisAI EP
- o load seguinte falhou com `com.ryzenai:MatMulNBits(-1) is not a registered function/op`
- apontar `custom_ops_library` para `libonnxruntime_vitis_ai_custom_ops.so` nao resolveu: o dominio `com.ryzenai` nao e registrado por essa lib
- inspecao do `model_jit.onnx` do Qwen3 mostra 332 nodes, dos quais 282 (85%) usam ops no dominio `com.ryzenai`: `MatMulNBits(121)`, `SLRN(80)`, `GQO(40)`, `SSMLP(40)`, `CastAvx(1)`
- esses ops sao proprietarios do EP `RyzenAI` da versao 1.7 e nao existem em nenhuma lib da tree 1.4.0
- conclusao: o modelo `Qwen3-14B-onnx-ryzenai-1.7-hybrid` e **incompativel** com o runtime 1.4.0; o gap nao e so versao do OGA, e o modelo inteiro depender de custom ops que so existem no EP RyzenAI 1.7

Probes de VAIML para o Whisper decoder:

- a `libvaip-pass_vaiml_partition.so` real (3.3 MB) do `npu-llm` foi copiada para `ryzen14`, substituindo o symlink-stub de 33 KB
- as 5 libs auxiliares VAIML tambem foram copiadas: `vaiml_custom_op`, `vaiml_mlopslib`, `vaiml_remove_dynamic_nodes`, `vaiml_remove_isolated_subgraph`, `vaiml_shape_infer`
- o encoder XINT8 continua funcionando com NPU 416 / CPU 122 — nao regrediu
- o decoder XINT8 continua em CPU 934 mesmo com todas as libs VAIML reais presentes
- o config DPU/DD (`vaip_config_npu_2_3.json`) nao ativa o caminho VAIML para o decoder
- misturar `LD_LIBRARY_PATH` com todas as libs do `npu-llm` prepended causa crash no encoder (`LogMessageFatal` em `vaip_dpu_custom_op`)

Probes de VAIML no container com tree npu-llm completa (2026-03-26):

- usando **toda** a tree de libs do npu-llm (sem misturar com ryzen14) + Python 3.10 nativo no container Ubuntu 22.04, o VAIML pass ativou de verdade
- config VAIML-only (`vaip_vaiml_only.json`) com apenas passes `init`, `vaiml`, `vaiml_partition`
- o decoder **XINT8** tentou particionar mas os subgrafos QDQ ficaram abaixo do threshold de 2% GOPs; resultado: `operators supported by VAIML: 0 (0.000%)`
- o decoder **FP32** teve resultado radicalmente diferente:
  - **235 de 236 ops (99.6%) suportadas pelo VAIML**
  - **35.84 de 35.84 GOPs (99.999%) suportadas**
  - 1 subgrafo cobrindo quase o modelo inteiro
  - porem a compilacao falhou com `sh: 1: aiecompiler: not found`
- o `aiecompiler` e parte do Vitis AI toolchain proprietario da AMD/Xilinx e nao existe no container nem no host

Leitura correta:

- o VAIML **consegue** particionar 99.6% do decoder Whisper FP32 para NPU
- o bloqueio final do decoder na NPU e a ausencia do `aiecompiler` para gerar o executavel AIE
- o decoder XINT8 nao e viavel pelo caminho VAIML; o caminho correto e FP32 → VAIML → BF16 no NPU
- o config DPU/DD (`vaip_config_npu_2_3.json`) ja inclui `vaiml` e `vaiml_partition` como passes habilitados, mas nao chega a ativa-los efetivamente no host por falta de dependencias
- misturar libs ryzen14 com npu-llm causa crash por ABI incompativel; usar tree npu-llm completa no container funciona
- em `2026-03-26` o `aiecompiler` foi localizado no venv 1.4.0 (`venv/bin/aiecompiler`, versao 2024.2.0)
- a compilacao exigia: locale `en_US.UTF-8`, symlinks de `aie_api/*` para `include/`, kernel headers (`linux-libc-dev`)
- apos corrigir os include paths e instalar dependencias, o decoder FP32 **compilou com sucesso**:
  - `flexml-compile Finished with status 0` (ambas fases)
  - **235 de 236 ops (99.6%) offloaded via VAIML**
  - **35.84 GOPs (99.999%) na NPU**
  - `[Vitis AI EP] No. of Operators :   CPU     1  VAIML   235`
  - `subgraphStat: [{'device': 'VAIML', 'count': 1}]`
  - compilacao levou ~365 segundos
- este e o primeiro offload de decoder transformer completo para NPU neste host
- a investigacao do build from source do ORT+VitisAI EP mostrou que o build falha em GCC 13/14 (issue microsoft/onnxruntime#27097), inviabilizando Fedora 43

### LLM GPU generico e DeepSeek R1

Foi criada uma infra generica de runner GPU que suporta modelos alem do Qwen3.

Novos artefatos:

- `tools/run_llm_gpu_transformers.py` — runner Python generico com suporte a GPTQ e BNB4
- `tools/run_llm_gpu_container.sh` — container runner generico
- `tools/run_llm_gpu_measured.sh` — medicao com monitor APU (generalizado do Qwen3)
- `tools/run_deepseek_r1_70b_gpu.sh` — wrapper dedicado para DeepSeek R1 70B

Descoberta importante: `DeepSeek-R1-Distill-Qwen-72B` nao existe. A serie Qwen distillada vai ate 32B. O modelo 70B usa base Llama: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`.

Para Q4 GPTQ (~37-42 GB VRAM), o candidato seria `empirischtech/DeepSeek-R1-Distill-Llama-70B-gptq-4bit`. O teste GPTQ no container ROCm falhou por problemas de build do `auto-gptq` e `gptqmodel` no ambiente ROCm. A alternativa mais pratica para 70B em GPU seria `ollama` ou `llama.cpp` com GGUF Q4.

### Qwen3 no GPU do host

Tambem foi feita uma rodada real do caminho GPU-only para `Qwen3` em container `ROCm/PyTorch`.

Estado observado agora no host Linux:

- o kernel expõe `/dev/kfd` e `/dev/dri/renderD128`
- o kernel `amdgpu` confirma `98304M` de VRAM na iGPU
- o host ainda nao tem `rocminfo`, `rocm-smi`, `hipcc` nem stack ROCm user-space em `PATH`
- mas o container `qwen3-gpu-pytorch` com `docker.io/rocm/pytorch:latest` ja foi validado com acesso real a GPU
- `Qwen/Qwen3-4B` roda de verdade nessa trilha GPU-only
- no split antigo `96 GB iGPU / 32 GB CPU`, `Qwen/Qwen3-32B` morria por `OOM` global do kernel em `2026-03-26 12:56:09`
- no split antigo `96 GB iGPU / 32 GB CPU`, `Qwen/Qwen3-14B` ficava preso em swap pesada na faixa `409/443`
- depois do rebalanceamento da BIOS para `64 GB iGPU / 64 GB CPU`, `Qwen/Qwen3-14B` e `Qwen/Qwen3-32B` passaram a carregar e gerar de verdade na iGPU

Medições úteis desta rodada:

- `Qwen3-4B`: load em cerca de `2.50 s`, generation em cerca de `2.84 s`, pico de `9.38 GiB` de VRAM, `87%` de `gpu_busy_percent`, `37.02 W`
- `Qwen3-32B` em `96/32`: pico de `62.8 GiB` de VRAM, `25.15 GiB` de RSS, `38.06 W`; morto por `OOM`
- `Qwen3-14B` em `96/32`: pico de `29.28 GiB` de VRAM, `26.34 GiB` de RSS, `100%` de `gpu_busy_percent`, `33.02 W`; sem `OOM`, mas operacionalmente impraticavel por swap
- `Qwen3-14B` em `64/64`: load em cerca de `12.10 s`, generation em cerca de `7.55 s`, pico de `28.96 GiB` de VRAM, `99%` de `gpu_busy_percent`, `34.09 W`
- `Qwen3-32B` em `64/64`: load em cerca de `85.54 s`, generation em cerca de `13.12 s`, pico de `58.03 GiB` de VRAM, `98%` de `gpu_busy_percent`, `37.02 W`

Leitura correta:

- o hardware do GPU esta visivel no kernel
- a trilha `Qwen3` em GPU ja esta pronta via container, mesmo sem stack ROCm no host
- o gargalo real do `Qwen3` grande neste host nao era a VRAM da iGPU
- o gargalo real era a RAM restante do lado CPU quando a BIOS estava em `96/32`
- com o split atual `64/64`, `4B`, `14B` e `32B` ficam viaveis na trilha GPU-only em container

### Whisper encoder na NPU via XINT8

Nesta retomada, o primeiro offload de transformer para a NPU foi provado no host Linux.

Estado validado agora:

- foi baixado o modelo `amd/whisper-tiny-en-onnx-npu` do HuggingFace
- o encoder é FP32 com shapes estáticos `[1, 80, 3000]` → `[1, 1500, 384]`, 135 nós, opset 20
- o encoder foi quantizado com `Quark XINT8` no container `vitis-ubuntu22-hub`
- a quantização atual do encoder deixou de usar ruído gaussiano aleatório e passou a usar calibração speech-like via `WhisperFeatureExtractor` sobre audio real do hub
- o modelo XINT8 resultante tem 7.9 MB (de 32 MB original)
- o probe nativo no host rodou com sucesso usando o config `vaip_config_npu_2_3.json` (DPU/DD)
- o relatório do EP mostra `NPU 416 / CPU 122`
- os ops na NPU incluem: `Gemm(12)`, `MatMul(4)`, `Softmax(4)`, `Add(17)`, `Mul(8)`, `Reshape(48)`
- os ops em CPU incluem: `LayerNormalization(9)`, `Gelu(6)`, `Conv(2)`
- `subgraphStat` marca `Actually running on NPU`

Comparação com resultados anteriores:

- `ResNet18 XINT8`: NPU 164 / CPU 2 (98.8% na NPU)
- `Whisper encoder XINT8`: NPU 416 / CPU 122 (77.3% na NPU)

Leitura correta de `NPU 416 / CPU 122`:

- isso nao e benchmark de tempo; isso e particao de operadores por device no relatorio do EP
- neste encoder, `416` ops foram para a NPU e `122` ficaram na CPU
- isso prova offload real, mas nao prova speedup por si so

Medição operacional rapida em `2026-03-26` com `runtime/whisper/sample_hf_1.flac` e o mesmo `tiny_en_encoder_xint8.onnx`:

- CPU warm, com sessao reutilizada: cerca de `0.061 s` por inferencia
- NPU warm, no runner nativo atual do hub com cache quente: cerca de `0.26 s` a `0.32 s` por inferencia
- NPU cold na primeira execucao: cerca de `7.75 s` por causa de compile/cache

Conclusão pratica desta medicao:

- hoje, neste encoder pequeno, a CPU ainda esta mais rapida que a trilha NPU usada pelo hub
- o ganho real desta trilha hoje e validacao de offload, nao ganho de latencia
- isso nao invalida a NPU; apenas mostra que, no estado atual do runtime e deste modelo, offload e speedup ainda nao sao a mesma coisa

Descobertas importantes desta rodada:

- o caminho VAIML (BF16 para transformers) nao funciona nesta tree Linux porque `libvaip-pass_vaiml_partition.so` nao existe; apenas `libvaip-pass_vaiml.so` existe e nao faz partição
- o caminho que funciona é `XINT8 via Quark` + config DPU/DD (`vaip_config_npu_2_3.json`)
- os modelos pre-built da AMD no HuggingFace (`amd/whisper-*-onnx-npu`) sao FP32 e esperam o caminho VAIML; sem esse plugin, nao vao para NPU
- a estratégia correta nesta tree é quantizar primeiro com Quark XINT8 e usar o config DPU/DD

### Whisper decoder XINT8 no host Linux

Tambem foi exercitado o decoder do Whisper no mesmo host Linux.

Estado validado agora:

- o arquivo `runtime/whisper/models/tiny_en_decoder.onnx` foi materializado com 117 MB
- o decoder foi quantizado com `Quark XINT8` para `runtime/whisper/models/tiny_en_decoder_xint8.onnx`
- o modelo XINT8 resultante tem 48 MB
- foi criado `tools/run_whisper_decoder_xint8_probe.sh` para rodar o decoder com `2` inputs (`x` e `xa`)
- foi criado `tools/probe_whisper_decoder.c` para ter um caminho C nativo dedicado a decoder multi-input
- o probe nativo do decoder sobe a sessao, roda inferencia e gera cache no host Linux
- `runtime/whisper/cache/xint8_decoder/whisper_dec_xint8/vitisai_ep_report.json` mostra `CPU 934`
- esse relatorio nao gera `subgraphStat`, entao hoje nao existe prova de particao real para NPU no decoder
- foi criado `tools/run_whisper_npu_transcribe.py` para costurar preprocessamento, encoder e decoder em uma trilha unica
- configs-alvo extras para transformer/DD foram testados em `runtime/whisper/configs`, mas nenhum fez o decoder sair de `CPU only`
- nesta tree Linux faltam componentes usados por essas variantes, como `libvaip-pass_level1_dd.so` e o pass `fuse_dynamic_dispatch_pss_pst`
- o submodelo extraido `tiny_en_decoder_body_xint8.onnx` tambem foi testado e continua inteiro em CPU (`CPU 921`)

Leitura correta desta rodada:

- a quantizacao do decoder ja foi feita
- o suporte basico a `2` inputs ja deixou de ser o bloqueio principal
- o bloqueio agora e fazer o decoder sair de `CPU only`
- sem isso, ainda nao existe prova final de Whisper completo na NPU
- `CPU 934` significa que os `934` ops do decoder ficaram inteiros na CPU; isso e zero offload, nao apenas offload parcial fraco

### Whisper híbrido ponta a ponta com áudio real

Tambem foi fechada nesta retomada uma trilha hibrida de transcricao real.

Estado validado agora:

- foi criado `tools/whisper_encode_dump.c` para rodar o encoder XINT8 pela C API nativa e despejar o tensor `[1, 1500, 384]`
- foi criado `tools/run_whisper_hybrid_transcribe.py` para usar encoder na NPU e decoder FP32 em CPU dentro do container
- foi criado `tools/run_whisper_full_hybrid.sh` como wrapper reproduzivel no host
- o prefixo correto para `openai/whisper-tiny.en` nesta trilha deixou de ser o prompt longo `[50257, 50258, 50358, 50362]` e passou a ser `tokenizer.prefix_tokens`, isto e `[50257, 50362]`
- com o prompt antigo, o decoder colapsava em `EOT` imediato; com o prefixo compacto, a trilha inteira passou a gerar texto
- o encoder continua provando `NPU 416 / CPU 122` no mesmo runner hibrido
- o melhor resultado validado hoje para `runtime/whisper/sample_hf_1.flac` e `"I'm"`
- o mesmo encoder XINT8 gera a mesma saida parcial em CPU, entao o gap atual deixou de ser o runtime da NPU e passou a ser a fidelidade do encoder quantizado
- uma segunda calibracao mais ampla com `6` arquivos de fala foi testada e descartada porque regrediu o resultado para `"I"`

Leitura correta desta rodada:

- a trilha ponta a ponta com audio real deixou de ser hipotese e virou fato reproduzivel
- o bloqueio principal do Whisper completo deixou de ser `EOT` imediato por prompt errado
- o bloqueio atual do pipeline hibrido e precisao do encoder `XINT8`
- o bloqueio atual do pipeline `Whisper completo na NPU` continua sendo duplo: precisao do encoder `XINT8` e decoder ainda `CPU only`

Investigação sobre OGA para LLM:

- AMD nao tem instalador Linux oficial para Ryzen AI Software (apenas Windows MSI)
- a doc `llm_linux.html` assume que voce já tem `ryzen_ai-1.6.1/venv` mas nao explica como obter no Linux
- os binários proprietários `libonnx_custom_ops.so`, `libryzen_mm.so`, `model_benchmark` nao existem em nenhum repo público
- o OGA genérico do PyPI (0.9.2) estaticamente linka ORT sem VitisAI EP
- o onnxruntime-genai v0.12.0 (fev 2026) mergeou suporte VitisAI EP + RyzenAI EP, mas nao existe wheel pré-compilada para Linux
- alternativa promissora para LLM: `FastFlowLM` (runtime NPU-first, suporte Linux desde março 2026)

## Situação real neste momento

- a NPU está visível no kernel
- a stack Linux da AMD foi montada localmente com progresso real
- já existe prova de sessão e inferência mínima também no host Linux, sem container
- já existe prova de sessão e inferência mínima em container `Ubuntu 22.04`
- já existe prova forte e reproduzível de offload real para NPU no host Linux com CNN (ResNet18)
- já existe prova forte e reproduzível de offload de transformer para NPU no host Linux (Whisper encoder)
- ja existe decoder Whisper XINT8 quantizado e inferindo no host Linux, mas ainda inteiro em CPU
- ja existe um pipeline hibrido de transcricao Whisper com audio real, encoder na NPU e decoder em CPU, mas o melhor resultado atual ainda e parcial (`"I'm"`)
- já existe prova de LLM pequena rodando nativamente no host Linux, mas em CPU
- ainda nao existe benchmark válido de Whisper completo com transcricao correta
- ainda nao existe prova final de LLM usando NPU
- o hardware local foi confirmado como `STX/KRK`, nao `PHX/HPT`
- o primeiro caminho validado de offload real para CNN é `resnet18_xint8_quark.onnx`
- o segundo caminho validado de offload real para transformer é `tiny_en_encoder_xint8.onnx`
- o runtime `OGA` ainda nao está materializado localmente em Linux
- o plugin `libvaip-pass_vaiml_partition.so` nao existe nesta tree; apenas o caminho XINT8+DPU funciona

## Prioridade imediata

Os próximos passos do hub devem ser:

1. fechar `OGA Linux` primeiro com `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
2. subir `Qwen3-14B-onnx-ryzenai-1.7-hybrid` pelo wrapper `tools/run_qwen3_14b_hybrid.sh`
3. materializar a stack `ROCm/PyTorch` do host para abrir a frente de `Qwen3` grande em GPU
4. manter `ResNet18 XINT8` e `Whisper encoder XINT8` como baselines que nao podem regredir

## Diretórios importantes

- `/var/home/mariostjr/amd-rai-linux/ryzen14`
- `/var/home/mariostjr/xdna-driver`
- `/var/home/mariostjr/xrt-ve2`
- `/var/home/mariostjr/LIRA`
- `/var/home/mariostjr/ryzen-ai-venv`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/llm_linux`

## Artefatos novos do hub

- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/probe_vitisai.c`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/make_minimal_identity_model.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_in_container.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_probe_native.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/make_probe_cnn_model.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_vitisai_partition_probe_native.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/make_resnet18_xint8_quark_model.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_resnet18_xint8_quark_probe.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/patch_oga_linux_model.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_oga_llm_linux.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/test_qwen_onnx.mjs`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/configs/vitisai_probe_bf16.json`
- `/var/home/mariostjr/Documents/hubs/NPUamd/third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/models/resnet18_fp32_opset13.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/models/resnet18_xint8_quark.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22/cache/resnet18_xint8_quark/vitisai_ep_report.json`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/quantize_whisper_encoder_xint8.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_encoder_probe.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_encoder_xint8_probe.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/quantize_whisper_decoder_xint8.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/probe_whisper_decoder.c`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_decoder_xint8_probe.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/extract_whisper_decoder_body.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/make_vaip_target_config.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/whisper_encode_dump.c`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_hybrid_transcribe.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_full_hybrid.sh`
- `/var/home/mariostjr/Documents/hubs/NPUamd/tools/run_whisper_npu_transcribe.py`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_encoder.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_encoder_xint8.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_decoder.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_decoder_xint8.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_decoder_body_xint8.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/cache/tiny_en_encoder_xint8/whisper_tiny_en_encoder_xint8/vitisai_ep_report.json`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/cache/xint8_decoder/whisper_dec_xint8/vitisai_ep_report.json`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/sample_hf_1.flac`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/whisper_hello.wav`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/calib`

## LTX-2.3 Video Generation on Radeon 8060S (gfx1151)

Data: 2026-04-01

### O que é

LTX-2.3 é um modelo DiT de 22B parâmetros da Lightricks para geração de vídeo+áudio a partir de texto.
Repositório: `https://github.com/Lightricks/LTX-2`
Modelo: `https://huggingface.co/Lightricks/LTX-2.3`

### Instalação

Localização: `/home/mariostjr/Documentos/hubs/LTX-2/`

Stack:
- Python 3.12.13 (via uv)
- PyTorch 2.12.0.dev20260328+rocm7.2 (nightly, único que funciona com gfx1151)
- ROCm 7.2.0 (sistema)
- transformers 4.52.4 (5.x quebra API do Gemma3)
- torchaudio 2.11.0.dev20260330+rocm7.2

Modelos descarregados em `LTX-2/models/`:
- `ltx-2.3-22b-distilled.safetensors` (43GB) - modelo principal
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (950MB) - upscaler 2x
- `ltx-2.3-22b-distilled-lora-384.safetensors` (7.1GB) - LoRA distilled
- `gemma3/` (23GB) - text encoder Gemma 3 12B (google/gemma-3-12b-it-qat-q4_0-unquantized)

### Workarounds para ROCm gfx1151

1. **LD_PRELOAD obrigatório**: `LD_PRELOAD="/usr/lib/libpthread.so.0 /opt/rocm/lib/librocprofiler-sdk.so.1"`
   - Sem isto, torch crasha ao importar (conflito entre libs ROCm 7.2 do sistema e ROCm bundled no wheel)

2. **TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1**: ativa attention experimental para gfx1151

3. **Tiling agressivo no VAE decode**: sem tiling, o VAE decoder crasha o driver ROCm ao processar vídeos com muitos frames.
   Solução: tiles espaciais de 256px (overlap 64px) + temporais de 16 frames (overlap 8 frames)

4. **transformers pinado a 4.52.x**: versões 5.x mudaram a API do Gemma3 (removeram `rope_local_base_freq`, `rotary_emb_local`)

### Scripts de execução

- `LTX-2/run_ltx_distilled.py` - Pipeline distilled (8+3 steps, mais rápido) com GPU tiling
- `LTX-2/run_ltx.py` - Pipeline one-stage (30 steps) com CPU decode fallback

### Comando para gerar vídeo

```bash
cd /home/mariostjr/Documentos/hubs/LTX-2
LD_PRELOAD="/usr/lib/libpthread.so.0 /opt/rocm/lib/librocprofiler-sdk.so.1" \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
.venv/bin/python run_ltx_distilled.py \
  --distilled-checkpoint-path models/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root models/gemma3 \
  --lora models/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --prompt "YOUR PROMPT" \
  --output-path output.mp4 \
  --num-frames 249 \
  --height 512 --width 768
```

### Benchmarks (Radeon 8060S, 64GB VRAM, ROCm 7.2)

| Vídeo | Stage 1 | Stage 2 | VAE Decode | Total |
|-------|---------|---------|------------|-------|
| 1s (25 frames, 512x768) | 13s | 13s | 7s (GPU tiled) | ~33s |
| 10s (249 frames, 512x768) | 76s | 4min03s | 3min35s (GPU tiled) | ~8min |

Pipeline distilled: 8 steps (stage 1) + 3 steps (stage 2) + tiled VAE decode.
Resolução de saída: 1024x1536 (upsampled 2x pelo spatial upscaler).

### Limitações conhecidas

- gfx1151 é experimental no ROCm/PyTorch - pode crashar com operações GPU muito longas
- Sem tiling, vídeos >1s crasham o driver HSA
- O nightly do PyTorch é instável - pode quebrar com atualizações
- MIOpen gera muitos warnings (inofensivos) na primeira execução
