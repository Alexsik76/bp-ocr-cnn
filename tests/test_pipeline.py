from bpocr.pipeline import (
    assemble_number,
    class_agnostic_nms,
    group_into_rows,
    padded_box,
)


def box(cls, cx, cy, h=10.0, w=6.0, conf=0.9):
    return {"cls": cls, "conf": conf, "cx": cx, "cy": cy, "h": h,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2}


def test_group_into_rows_three_rows_uneven_counts():
    boxes = [
        box(1, 60, 20), box(2, 40, 21), box(5, 20, 19),  # row 1: 3 digits
        box(7, 40, 100), box(4, 20, 101),                # row 2: 2 digits
        box(6, 20, 180),                                 # row 3: 1 digit
    ]
    rows = group_into_rows(boxes)
    assert [[b["cls"] for b in r] for r in rows] == [[5, 2, 1], [4, 7], [6]]


def test_group_into_rows_fewer_than_three_boxes():
    rows = group_into_rows([box(1, 10, 50), box(2, 10, 10)])
    assert [[b["cls"] for b in r] for r in rows] == [[2], [1]]


def test_group_into_rows_empty():
    assert group_into_rows([]) == []


def test_assemble_number():
    row = [box(1, 10, 0), box(2, 20, 0), box(5, 30, 0)]
    assert assemble_number(row) == 125
    assert assemble_number([box(0, 10, 0), box(7, 20, 0)]) == 7
    assert assemble_number([]) is None


def test_class_agnostic_nms_drops_overlap_of_different_class():
    strong = box(2, 50, 50, conf=0.9)
    weak_dup = box(3, 51, 50, conf=0.5)
    other = box(8, 200, 50, conf=0.4)
    kept = class_agnostic_nms([weak_dup, other, strong], 0.4)
    assert [b["cls"] for b in kept] == [2, 8]


def test_class_agnostic_nms_empty():
    assert class_agnostic_nms([]) == []


def test_padded_box_adds_padding():
    assert padded_box((100, 200, 200, 400), 1000, 1000, 0.03) == (97, 194, 203, 406)


def test_padded_box_clamps_to_image():
    assert padded_box((0, 5, 995, 100), 1000, 102, 0.05) == (0, 1, 1000, 102)
