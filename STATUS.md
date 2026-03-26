# Status Atual

## Objetivo

Fazer a NPU AMD funcionar no Linux host atual, com foco prático em:

- Whisper
- LLM local

## Host

- Usuário: `mariostjr`
- Sistema: Bazzite/Fedora imutável
- Modelo de gestão: `rpm-ostree`
- Data deste registro: `2026-03-24`
- Plataforma NPU confirmada agora: `STX/KRK`
- PCI ID da NPU: `1022:17f0`
- Revisão observada da NPU: `rev 11`

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

Leitura correta:

- o passo 2 ainda nao está bloqueado por hardware
- ele está bloqueado por ausência de runtime/artefatos corretos no Linux local
- nao se deve presumir que a presença dos exemplos Python da AMD ou da wheel genérica `onnxruntime-genai` equivale à presença do runtime Linux oficial da AMD

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

### Whisper encoder na NPU via XINT8

Nesta retomada, o primeiro offload de transformer para a NPU foi provado no host Linux.

Estado validado agora:

- foi baixado o modelo `amd/whisper-tiny-en-onnx-npu` do HuggingFace
- o encoder é FP32 com shapes estáticos `[1, 80, 3000]` → `[1, 1500, 384]`, 135 nós, opset 20
- o encoder foi quantizado com `Quark XINT8` no container `vitis-ubuntu22-hub`
- o modelo XINT8 resultante tem 7.9 MB (de 32 MB original)
- o probe nativo no host rodou com sucesso usando o config `vaip_config_npu_2_3.json` (DPU/DD)
- o relatório do EP mostra `NPU 416 / CPU 122`
- os ops na NPU incluem: `Gemm(12)`, `MatMul(4)`, `Softmax(4)`, `Add(17)`, `Mul(8)`, `Reshape(48)`
- os ops em CPU incluem: `LayerNormalization(9)`, `Gelu(6)`, `Conv(2)`
- `subgraphStat` marca `Actually running on NPU`

Comparação com resultados anteriores:

- `ResNet18 XINT8`: NPU 164 / CPU 2 (98.8% na NPU)
- `Whisper encoder XINT8`: NPU 416 / CPU 122 (77.3% na NPU)

Descobertas importantes desta rodada:

- o caminho VAIML (BF16 para transformers) nao funciona nesta tree Linux porque `libvaip-pass_vaiml_partition.so` nao existe; apenas `libvaip-pass_vaiml.so` existe e nao faz partição
- o caminho que funciona é `XINT8 via Quark` + config DPU/DD (`vaip_config_npu_2_3.json`)
- os modelos pre-built da AMD no HuggingFace (`amd/whisper-*-onnx-npu`) sao FP32 e esperam o caminho VAIML; sem esse plugin, nao vao para NPU
- a estratégia correta nesta tree é quantizar primeiro com Quark XINT8 e usar o config DPU/DD

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
- já existe prova de LLM pequena rodando nativamente no host Linux, mas em CPU
- ainda nao existe benchmark válido de Whisper completo (encoder + decoder + transcrição real)
- ainda nao existe prova final de LLM usando NPU
- o hardware local foi confirmado como `STX/KRK`, nao `PHX/HPT`
- o primeiro caminho validado de offload real para CNN é `resnet18_xint8_quark.onnx`
- o segundo caminho validado de offload real para transformer é `tiny_en_encoder_xint8.onnx`
- o runtime `OGA` ainda nao está materializado localmente em Linux
- o plugin `libvaip-pass_vaiml_partition.so` nao existe nesta tree; apenas o caminho XINT8+DPU funciona

## Prioridade imediata

Os próximos passos do hub devem ser:

1. quantizar e testar o decoder Whisper com XINT8 para completar a prova de transcrição real na NPU
2. montar pipeline completo de transcrição Whisper (encoder + decoder + tokenizer + áudio)
3. investigar FastFlowLM como caminho alternativo para LLM na NPU no Linux
4. manter baselines reproduzíveis para nao regredir

## Diretórios importantes

- `/var/home/mariostjr/amd-rai-linux/ryzen14`
- `/var/home/mariostjr/xdna-driver`
- `/var/home/mariostjr/xrt-ve2`
- `/var/home/mariostjr/LIRA`
- `/var/home/mariostjr/ryzen-ai-venv`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/ubuntu22`
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
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_encoder.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_encoder_xint8.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/models/tiny_en_decoder.onnx`
- `/var/home/mariostjr/Documents/hubs/NPUamd/runtime/whisper/cache/tiny_en_encoder_xint8/whisper_tiny_en_encoder_xint8/vitisai_ep_report.json`
