"""Hy4-flash tokenizer oracle: token ids of llama.cpp's llama-tokenize (vocab-only load, CPU) on a fixed corpus, written to tests/data/hy4_tok_oracle.json.

    python tools/hy4_tok_oracle.py --bin <llama.cpp>/build-cpu/bin/llama-tokenize --gguf <Hy4-flash ...-00001-of-0000N.gguf> [--out tests/data/hy4_tok_oracle.json]

Every string runs twice: the default (control tokens in the text are one id, what a chat server feeds) and `--no-parse-special` (control tokens are text;
llama.cpp still matches USER_DEFINED tokens). The corpus is built here, deterministic (hand cases + seeded fuzz), so a later run gives the same file."""
import argparse
import json
import os
import random
import subprocess
import tempfile

SPECIAL = ["<｜hy_start:6124c78e｜>", "<｜hy_middle:6124c78e｜>", "<｜hy_pad:6124c78e｜>", "<｜hy_fim▁hole｜>", "<｜hy_fim▁begin｜>", "<｜hy_fim▁end｜>",
           "<｜hy_place▁holder▁no▁2｜>", "<｜hy_place▁holder▁no▁120006｜>"]

HAND = [
    # English, numbers, punctuation
    "Hello world", "Hello, world!", " leading space", "trailing space ", "  two leading", "double  inner  spaces", "tab\tseparated\tvalues",
    "line1\nline2", "line1\r\nline2", "para\n\n\npara", "\n", "\n\n", " ", "   ", "\t", "\r", "\r\n\r\n", "", "a", "I'm here, you're there, it's fine.",
    "don't won't can't shouldn't've", "1", "12", "123", "1234", "1234567890", "3.14159", "-42", "1,000,000", "2025-11-18T10:38:00Z", "v1.2.3-rc4",
    "$19.99", "100%", "#hashtag @mention", "e.g. i.e. etc.", "...", "!!!???", "(parens) [brackets] {braces}", "a/b\\c|d", "<html><body>x</body></html>",
    "https://example.com/path?q=1&r=2#frag", "user@example.org", "CamelCaseIdentifier", "snake_case_identifier", "SCREAMING_CASE", "x^2 + y^2 = z^2",
    "The quick brown fox jumps over the lazy dog.", "A\u00a0non-breaking space", "em\u2014dash and en\u2013dash", "\u201csmart quotes\u201d \u2018single\u2019",
    "zero\u200bwidth", "ideographic\u3000space", "word\u2028line separator", "soft\u00adhyphen", "ﬁ ligature", "Ⅻ roman", "½ ¾ ²³", "°C ±5",
    # French
    "Bonjour, comment ça va ?", "L'été dernier, j'ai mangé une crème brûlée.", "Où est la bibliothèque ?", "Ça coûte 12,50 € !", "« guillemets français »",
    "Les élèves s'inquiètent : « Est-ce qu'il pleuvra demain ? »", "œuvre, cœur, Æsop", "À bientôt, Éloïse !", "naïve façade déjà-vu", "aujourd'hui",
    "Je m'appelle Yves et j'habite à Saint-Étienne.", "e\u0301 combining acute", "Ame\u0301lie (NFD)", "Amélie (NFC)",
    # Chinese / Japanese / Korean
    "你好，世界！", "我爱北京天安门。", "今天天气很好，我们去公园散步吧。", "机器学习是人工智能的一个分支。", "中文English混合Mixed", "数字123和字母abc",
    "龥龦鿿", "㐀㐁 extension A", "𠀀𠀁 extension B", "繁體中文測試", "「引号」『书名号』", "《三体》是刘慈欣写的。", "一二三四五六七八九十",
    "こんにちは世界", "ひらがなカタカナ漢字", "東京タワーへ行きました。", "ｶﾀｶﾅ halfwidth", "ラーメン・ヴィ", "안녕하세요 세계", "한국어와 中文 그리고 日本語",
    "中\n文", "中 文", "，。！？；：", "全角ＡＢＣ１２３",
    # Other scripts
    "Привет, мир!", "Γειά σου Κόσμε", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก", "Xin chào thế giới", "Tiếng Việt có dấu",
    "ʻokina Hawaiʻi", "ß ẞ Straße", "İstanbul ıi", "Ꭰ Cherokee", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math", "𝟏𝟐𝟑 bold digits", "١٢٣ arabic digits", "৫৬ bengali digits",
    # Emojis
    "🙂", "😀😃😄", "👍🏽", "👨‍👩‍👧‍👦 family", "🇫🇷🇨🇳🇯🇵", "❤️", "⭐️✨", "🏳️‍🌈", "I ❤️ Paris 🗼", "emoji🙂inside", "🤖 + 🧠 = 🚀", "©®™", "☃︎ ☃️",
    "1️⃣ keycap", "🧑🏿‍💻", "😂😂😂😂😂", "→ ← ↑ ↓ ⇒", "∑∫∂√∞ ≠ ≤ ≥", "♠♥♦♣", "★☆",
    # Code
    "def f(x):\n    return x**2\n", "for i in range(10):\n\tprint(i)", "if (a && b || !c) { return -1; }", "int main(void) {\n  printf(\"%d\\n\", 42);\n}\n",
    "const x = () => { console.log(`tpl ${y}`); };", "SELECT * FROM users WHERE id = 1;", "{\"key\": [1, 2.5, true, null], \"s\": \"v\"}",
    "#include <stdio.h>", "import numpy as np\nimport mlx.core as mx", "x = a[1:3, ::2]", "    indented 4", "        indented 8", "\t\tindented tabs",
    "let mut v: Vec<u8> = Vec::new();", "fn main() -> Result<(), Box<dyn Error>> {}", "<?php echo $x; ?>", "a == b != c <= d >= e", "i++; --j; k += 1;",
    "// comment\n/* block */", "# python comment", "'''docstring'''", "r'\\d+\\s*'", "0x1F 0b1010 1e-9 3_000", "foo.bar.baz()", "obj->field", "std::vector<int>",
    "<div class=\"a\">中文</div>", "```python\nprint('hi')\n```", "| a | b |\n|---|---|\n| 1 | 2 |", "- item\n- item2\n  - nested", "**bold** _it_ `code`",
    "$ ls -la ~/dir | grep x > out.txt 2>&1", "\\n\\t escaped literal", "key=value; other=1", "<tool_call>{\"name\":\"f\"}</tool_call>",
    # Special tokens embedded
    SPECIAL[0] + "user\nhi", "a" + SPECIAL[1] + "b", SPECIAL[2] * 3, SPECIAL[3] + "def f():" + SPECIAL[4] + "pass" + SPECIAL[5], " " + SPECIAL[6] + " ",
    "x" + SPECIAL[7] + "\n", "<｜hy_start:6124c78e", "<｜not_a_token｜>", "<|im_start|>user", "<s></s>", SPECIAL[0] + SPECIAL[1],
    # Whitespace runs mixed with letters / punctuation
    "a  \n  b", "a \n\n b", "x   ", "   x", " .", ". ", "a.b", "a. b", " ?!", "end.\n", "end?\r\n", "foo   \nbar", "\n   \n", "a\t \tb", "--flag=1",
    "'quote", "\"dq\"", "(a)", "a's", "A's B'S", "word's.", ".hidden", ",comma", ";semi", "_under", "-dash", "+plus", "=eq", "@at", "~tilde",
]

POOLS = ["abcxyz", "ABCXYZ", "àéîõüçœ", "你好世界中文字", "ひらカタ", "한글", "0123456789", " ", "  ", "\t", "\n", "\r\n", ".,;:!?", "'\"`", "()[]{}<>",
         "-_+=*/\\|&^%$#@~", "🙂👍🏽🇫🇷❤️", "€£¥", "\u00a0\u3000", "e\u0301", "αβγ", "дж", "،؟", "→∑", "＿，。"]


def corpus() -> list:
    rng = random.Random(1118)
    fuzz = []
    while len(fuzz) < 150:
        s = "".join(rng.choice(rng.choice(POOLS)) for _ in range(rng.randint(1, 24)))
        if rng.random() < 0.15:
            s = s[:len(s) // 2] + rng.choice(SPECIAL) + s[len(s) // 2:]
        if s not in fuzz and s not in HAND:
            fuzz.append(s)
    return HAND + fuzz


def tokenize(binary: str, gguf: str, text: str, special: bool, tmp: str | None = None) -> list:
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False, dir=tmp) as fh:
        fh.write(text.encode("utf-8"))
        p = fh.name
    try:
        cmd = [binary, "-m", gguf, "-t", "4", "-f", p, "--no-escape", "--ids", "--log-disable"] + ([] if special else ["--no-parse-special"])
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    finally:
        os.unlink(p)
    return json.loads(out[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tmp", help="directory for the prompt files (default: the system temp dir)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tests", "data", "hy4_tok_oracle.json"))
    a = ap.parse_args()
    cases = [{"text": t, "ids": tokenize(a.bin, a.gguf, t, True, a.tmp), "ids_plain": tokenize(a.bin, a.gguf, t, False, a.tmp)} for t in corpus()]
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(a.bin)))), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"llama_cpp": commit, "gguf": os.path.basename(a.gguf), "n_vocab": 120832, "cases": cases}, fh, ensure_ascii=False, indent=0)
    print(len(cases), "cases ->", a.out)


if __name__ == "__main__":
    main()
