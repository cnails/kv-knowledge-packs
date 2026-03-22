"""
Exp 93b: KV Composition — Can independently-built KV caches be combined?

Tests whether KV caches built from separate paragraphs can be concatenated
and still enable multi-hop reasoning. This is a unique KV capability:
modular knowledge packs that compose like LEGO.

- KV Composed: build KV for para_1 and para_2 separately, concatenate
- KV Single: both paragraphs in one KV (current approach, upper bound)
- RAG top-1 and Prefix gold for comparison

N=200 bridge questions from HotpotQA.
"""
import torch, json, time, random, os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

device = torch.device('cuda')
MODEL_NAME = 'meta-llama/Llama-3.1-70B-Instruct'
HF_TOKEN = os.environ.get('HF_TOKEN', '')

print('Loading model...', flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map='auto', token=HF_TOKEN)
model.eval()
print(f'Model loaded in {time.time()-t0:.0f}s', flush=True)

with open('/root/hotpot_dev_distractor_v1.json') as f:
    hotpot = json.load(f)

def clone_kv(kv):
    c = DynamicCache()
    for li in range(len(kv.layers)):
        c.update(kv.layers[li].keys.clone(), kv.layers[li].values.clone(), li)
    return c

def build_kv(text, max_tok=6144):
    ids = tok.encode(text, add_special_tokens=False)[:max_tok]
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, use_cache=True)
    return out.past_key_values, len(ids)

def compose_kv(kv_list):
    """Concatenate multiple KV caches along sequence dimension."""
    composed = DynamicCache()
    n_layers = len(kv_list[0].layers)
    for li in range(n_layers):
        keys = torch.cat([kv.layers[li].keys for kv in kv_list], dim=2)
        vals = torch.cat([kv.layers[li].values for kv in kv_list], dim=2)
        composed.update(keys, vals, li)
    return composed

def gen_kv(query, kv, kv_len, max_tokens=50):
    sys_msg = 'Answer the question concisely. Give a short, direct answer.'
    msgs = [{'role':'system','content':sys_msg},{'role':'user','content':query}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    q_ids = tok.encode(prompt, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    am = torch.ones(1, kv_len+len(q_ids), device=device, dtype=torch.long)
    kv_c = clone_kv(kv)
    with torch.no_grad():
        out = model.generate(qt, past_key_values=kv_c, attention_mask=am,
                             max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()

def gen_prefix(query, facts_text, max_tokens=50):
    sys_msg = 'Answer the question concisely based on the provided context. Give a short, direct answer.'
    user_msg = 'Context:\n' + facts_text + '\n\nQuestion: ' + query
    msgs = [{'role':'system','content':sys_msg},{'role':'user','content':user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()

def check(pred, gold):
    p, g = pred.lower().strip(), gold.lower().strip()
    if g in p: return True
    pt, gt = set(p.split()), set(g.split())
    if gt and pt and len(pt & gt)/len(gt) >= 0.5: return True
    return False

# Same seed as other experiments
random.seed(42)
bridge_qs = [q for q in hotpot if q.get('type') == 'bridge']
test_qs = random.sample(bridge_qs, 200)
print(f'Testing {len(test_qs)} bridge questions', flush=True)

results = {
    'kv_single': [],      # Both paras in one KV (upper bound)
    'kv_composed': [],     # Two separate KVs concatenated
    'prefix_gold': [],     # Both paras in prompt
}

for i, item in enumerate(test_qs):
    question = item['question']
    gold_answer = item['answer']
    sup_titles = set(sf[0] for sf in item['supporting_facts'])

    gold_paras = []
    for title, sents in item['context']:
        joined = ' '.join(sents)
        text = title + ': ' + joined
        if title in sup_titles:
            gold_paras.append(text)

    if len(gold_paras) < 2:
        # Skip if less than 2 gold paragraphs
        for k in results:
            results[k].append(False)
        continue

    if i % 20 == 0:
        print(f'\n--- Progress: {i}/{len(test_qs)} ---', flush=True)
        for k, v in results.items():
            if v:
                print(f'    {k}: {sum(v)}/{len(v)} = {100*sum(v)/len(v):.1f}%', flush=True)

    # 1. KV Single (both paras together)
    try:
        kv, kv_len = build_kv(' '.join(gold_paras))
        ans = gen_kv(question, kv, kv_len)
        del kv
    except: ans = ''
    results['kv_single'].append(check(ans, gold_answer))

    # 2. KV Composed (separate KVs concatenated)
    try:
        kv1, kv_len1 = build_kv(gold_paras[0])
        kv2, kv_len2 = build_kv(gold_paras[1])
        composed = compose_kv([kv1, kv2])
        total_len = kv_len1 + kv_len2
        ans = gen_kv(question, composed, total_len)
        del kv1, kv2, composed
    except Exception as e:
        ans = ''
        if i < 3:
            print(f'  Compose error: {e}', flush=True)
    results['kv_composed'].append(check(ans, gold_answer))

    # 3. Prefix gold
    try:
        ans = gen_prefix(question, '\n'.join(gold_paras))
    except: ans = ''
    results['prefix_gold'].append(check(ans, gold_answer))

    if i < 3:
        print(f'  Q: {question}', flush=True)
        print(f'  Gold: {gold_answer}', flush=True)
        mk = lambda h: 'Y' if h else 'X'
        print(f'  KV single:   [{mk(results["kv_single"][-1])}]', flush=True)
        print(f'  KV composed: [{mk(results["kv_composed"][-1])}]', flush=True)
        print(f'  Prefix gold: [{mk(results["prefix_gold"][-1])}]', flush=True)

    if i % 10 == 0:
        torch.cuda.empty_cache()

print('\n' + '='*60, flush=True)
print('KV COMPOSITION RESULTS (N=200 bridge)', flush=True)
print('='*60, flush=True)
for method, hits in results.items():
    c = sum(hits)
    n = len(hits)
    print(f'  {method:>15s}: {c}/{n} = {100*c/n:.1f}%', flush=True)

gap = abs(100*sum(results['kv_single'])/len(results['kv_single']) - 100*sum(results['kv_composed'])/len(results['kv_composed']))
print(f'\n  KV Single - KV Composed gap: {gap:.1f}pp', flush=True)
if gap <= 3:
    print('  → Composition works! Independent KV caches compose like LEGO.', flush=True)
else:
    print('  → Composition degrades. Positional encoding matters.', flush=True)

# Overlap analysis
both = sum(1 for a,b in zip(results['kv_single'],results['kv_composed']) if a and b)
single_only = sum(1 for a,b in zip(results['kv_single'],results['kv_composed']) if a and not b)
composed_only = sum(1 for a,b in zip(results['kv_single'],results['kv_composed']) if not a and b)
neither = sum(1 for a,b in zip(results['kv_single'],results['kv_composed']) if not a and not b)
print(f'\n  Both correct: {both}', flush=True)
print(f'  Single only: {single_only} (composition hurt)', flush=True)
print(f'  Composed only: {composed_only} (composition helped)', flush=True)
print(f'  Neither: {neither}', flush=True)

print('\nDone.', flush=True)
