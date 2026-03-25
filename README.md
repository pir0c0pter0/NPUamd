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
- sim, ja existe prova forte de offload real para a NPU no host Linux
- nao, `Whisper` e `LLM` ainda nao estao fechados com prova final de NPU neste host
- o proximo alvo correto e `OGA Linux` com `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`

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

### 4. O que ainda nao vale como prova

Ainda nao vale como prova de NPU:

- baseline de `Qwen` em `transformers.js` no host, porque hoje ele roda em CPU
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
- `tools/run_oga_llm_linux.sh`
- `tools/patch_oga_linux_model.py`

## Melhor baseline hoje

Se eu precisasse retomar do zero com o menor risco tecnico, eu faria nesta ordem:

1. manter `resnet18_xint8_quark.onnx` como baseline reproduzivel do primeiro offload real
2. fechar `OGA` em Linux usando primeiro `amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu`
3. so depois subir para `Qwen` da colecao AMD `OGA/NPU`

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

O primeiro offload real para a NPU AMD no Linux host ja foi provado; agora o trabalho serio e fechar `OGA Linux` com `Phi-3.5` usando a trilha oficial da AMD, em vez de perder tempo com caminhos genericos que so mostram CPU.
