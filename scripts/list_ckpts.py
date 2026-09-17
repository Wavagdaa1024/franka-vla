from pathlib import Path
import torch
import json

res = []
for p in sorted(Path('outputs/checkpoints').glob('*/*.pt')):
    try:
        c = torch.load(p, map_location='cpu', weights_only=False)
        sd = c.get('state_dict', c.get('model_state_dict', c))
        is_lora = any('lora' in k.lower() for k in sd.keys()) if isinstance(sd, dict) else False
        res.append({
            'folder': p.parent.name,
            'filename': p.name,
            'size_mb': round(p.stat().st_size / (1024*1024), 2),
            'step': c.get('step', '?'),
            'loss': round(float(c.get('loss')), 5) if c.get('loss') is not None else None,
            'tune_vision': c.get('tune_vision', False),
            'vision_layers': c.get('vision_layers', 0),
            'is_lora': is_lora,
            'lang_rank': c.get('lang_rank', None),
            'expert_rank': c.get('expert_rank', None)
        })
    except Exception as e:
        res.append({'file': str(p), 'error': str(e)})

print(json.dumps(res, indent=2))
