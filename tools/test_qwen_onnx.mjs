import path from 'node:path';
import process from 'node:process';

import { env, pipeline } from '@huggingface/transformers';

const MODEL_ID = process.env.MODEL_ID || 'onnx-community/Qwen2.5-0.5B-Instruct';
const CACHE_DIR = process.env.HF_CACHE_DIR || path.resolve(process.cwd(), '.cache', 'huggingface');
const PROMPT = process.env.QWEN_PROMPT || 'Explique em duas frases o que ja funciona hoje da NPU AMD no Linux neste host.';
const MAX_NEW_TOKENS = Number(process.env.MAX_NEW_TOKENS || 48);

env.allowRemoteModels = true;
env.allowLocalModels = true;
env.cacheDir = CACHE_DIR;

console.log(`[INFO] model=${MODEL_ID}`);
console.log(`[INFO] cache=${CACHE_DIR}`);
console.log('[INFO] loading pipeline...');

const generator = await pipeline('text-generation', MODEL_ID, {
  device: 'cpu',
  dtype: 'q4'
});

const messages = [
  { role: 'system', content: 'Responda em portugues do Brasil e seja objetivo.' },
  { role: 'user', content: PROMPT }
];

console.log('[INFO] running generation...');
const output = await generator(messages, {
  max_new_tokens: MAX_NEW_TOKENS,
  do_sample: false,
  temperature: 0
});

const first = Array.isArray(output) ? output[0] : output;
const generatedText = Array.isArray(first?.generated_text)
  ? first.generated_text.at(-1)?.content ?? JSON.stringify(first.generated_text)
  : first?.generated_text ?? JSON.stringify(output);

console.log('[OK] generation completed');
console.log('--- PROMPT ---');
console.log(PROMPT);
console.log('--- OUTPUT ---');
console.log(generatedText);
