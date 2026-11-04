"""mu_keep: a pass whose sessions sit in slots that are not 0..S-1 maps rows to the same rings as the identity stack. The slot only names the buffer."""
import numpy as np
from kyojin_mlx.dsv4_core import Plan as Plan


def same(a, b):
    return np.array_equal(np.array(a), np.array(b))


def test_slots_change_the_buffer_not_the_rows():
    pos, counts = [10, 11, 12, 40, 41, 70], [3, 2, 1]
    base = Plan(pos, counts)
    for slots in ([1, 2, 3], [2, 0, 1], [3, 1, 2]):
        p = Plan(pos, counts, slots=slots, nslots=4)
        assert p.rank == base.rank == [0, 0, 0, 1, 1, 2]
        assert p.sess == [slots[s] for s in base.rank]
        assert same(p.first_arr(), base.first_arr())
        assert same(p.wmap(8), base.wmap(8))
        o, b = p.overlay(8), base.overlay(8)
        assert same(o.rows, b.rows) and same(o.slots, b.slots)


def test_plain_step_with_a_high_slot_does_not_index_past_first():
    p = Plan([5, 6], None, slots=[2, 3], nslots=4)
    assert p.wmap(4).shape[0] == 8
    assert same(p.first_arr(), [0, 1])
