"""transformers' {% generation %} tag (recent chat templates, e.g. Qwen3.8-27B-FP8): rendered as its body, same text as without the tag."""
from kyojin_mlx.serve import chat


def test_generation_tag_renders_its_body():
    with_tag = "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}<a>{{ m.content }}</a>{% endgeneration %}{% else %}<u>{{ m.content }}</u>{% endif %}{% endfor %}"
    plain = with_tag.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert chat._compiled(with_tag).render(messages=msgs) == chat._compiled(plain).render(messages=msgs) == "<u>hi</u><a>yo</a>"


def test_template_strftime_now_global():
    """transformers exposes strftime_now to chat templates (gpt-oss harmony template calls it for the current date)."""
    from kyojin_mlx.serve.chat import _template_env
    out = _template_env().from_string("{{ strftime_now('%Y') }}").render()
    assert out.isdigit() and len(out) == 4
