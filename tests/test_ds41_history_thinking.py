from kyojin_mlx.serve import chat, ds41_chat

TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
HISTORY = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
           {"role": "assistant", "content": "a", "reasoning_content": "why"}, {"role": "user", "content": "q2"}]
SUMMARY = HISTORY + [{"role": "assistant", "content": "b", "reasoning_content": "why2"}, {"role": "user", "content": "summarize"}]


def opts(**ctk):
    return ds41_chat.chat_options({"chat_template_kwargs": ctk})


def test_option_needs_thinking_off():
    assert opts(enable_thinking=False, keep_history_thinking=True).history_thinking
    assert not opts(keep_history_thinking=True).history_thinking
    assert not opts(enable_thinking=False).history_thinking


def test_summary_turn_keeps_the_session_prefix():
    warm = ds41_chat.render_chat(HISTORY, TOOLS)
    tail_line = ds41_chat.SYSTEM_SP + ds41_chat.effort_line(ds41_chat.resolve_effort(ds41_chat.DEFAULT_EFFORT))
    head = warm[:-len(ds41_chat.USER_SP + "q2" + tail_line + ds41_chat.ASSISTANT_SP + ds41_chat.THINK_OPEN)]
    keep = ds41_chat.render_chat(SUMMARY, TOOLS, thinking=False, history_thinking=True)
    assert keep.startswith(head) and "why" in keep
    assert keep.endswith(ds41_chat.ASSISTANT_SP + ds41_chat.THINK_OPEN + ds41_chat.THINK_CLOSE)
    assert not ds41_chat.render_chat(SUMMARY, TOOLS, thinking=False).startswith(head)


def test_rendered_answers_without_thinking_and_cuts_match():
    fam = chat.Ds41Family.__new__(chat.Ds41Family)
    fam.thinking, fam.effort = True, ds41_chat.DEFAULT_EFFORT
    body = {"messages": SUMMARY, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False, "keep_history_thinking": True}}
    o = ds41_chat.chat_options(body)
    prompt = ds41_chat.render_chat(SUMMARY, TOOLS, thinking=False, history_thinking=True)
    cuts = fam._cut_texts(body, TOOLS, o, prompt)
    assert len(cuts) == 2 and all(prompt.startswith(c) for c in cuts)
    assert cuts[-1] == prompt[:-len(ds41_chat.ASSISTANT_SP + ds41_chat.THINK_OPEN + ds41_chat.THINK_CLOSE)]


def test_effort_tail_keeps_the_prefix_when_effort_changes():
    a = ds41_chat.render_chat(HISTORY, TOOLS, reasoning_effort="medium", effort_placement="tail")
    b = ds41_chat.render_chat(HISTORY, TOOLS, reasoning_effort="max", effort_placement="tail")
    cut = a.rindex(ds41_chat.SYSTEM_SP + "Reasoning Effort")
    assert a[:cut] == b[:cut] and a != b
    h1 = ds41_chat.render_chat(HISTORY, TOOLS, reasoning_effort="medium", effort_placement="head")
    h2 = ds41_chat.render_chat(HISTORY, TOOLS, reasoning_effort="max", effort_placement="head")
    assert h1[:60] != h2[:60] or h1[60:200] != h2[60:200]


def test_effort_tail_cut_text_ends_before_the_effort_line():
    fam = chat.Ds41Family.__new__(chat.Ds41Family)
    body = {"messages": HISTORY, "tools": TOOLS, "reasoning_effort": "max"}
    o = ds41_chat.chat_options(body)
    prompt = ds41_chat.render_chat(HISTORY, TOOLS, reasoning_effort="max", effort_placement="tail")
    cuts = fam._cut_texts(body, TOOLS, o, prompt)
    assert cuts[-1].endswith("q2") and "Reasoning Effort" not in cuts[-1][cuts[-1].index("q2"):]
