"""tiny helper for the G4 check scripts: real-text token ids (wikitext slice) with the GLM tokenizer."""
import os


def ids_for(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/glm53flash_fp4local"))
    return tok.encode(open(os.path.join(os.path.expanduser(os.environ.get("G4_WORK", "~/g4_work")), "cut_text.txt")).read(), add_special_tokens=False)[:n]
