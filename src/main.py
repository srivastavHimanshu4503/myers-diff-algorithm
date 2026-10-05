"""Myers' O(ND) diff: line diff (Part A) and changed-character ranges (Part B).

The diff core operates on arbitrary hashable sequences. The CLI uses it for
line-level diffs over raw bytes and character-level diffs over strings.
"""

import sys


def read_lines(path):
    """Read a file as raw bytes and split it into logical lines."""
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")

    # A trailing newline does not represent an additional logical line.
    if lines and not lines[-1]:
        lines.pop()

    return lines


def middle_snake(A, B, Ar, Br, n, m):
    """Find the middle snake in the Myers edit graph.

    Returns:
        (sx, sy, ex, ey)

    representing the snake from (sx, sy) to (ex, ey).

    A/B and Ar/Br contain one sentinel element beyond their logical bounds.
    """
    delta = n - m
    odd = delta & 1

    max_d = (n + m + 1) // 2
    offset = max_d + 1
    size = 2 * max_d + 3

    # Furthest x reached on each diagonal.
    vf = [-1] * size
    vb = [-1] * size

    vf[offset + 1] = 0
    vb[offset + 1] = 0

    # Active diagonal boundaries.
    fs = fe = bs = be = 0

    for d in range(max_d + 1):
        # ---------------------------------------------------------------
        # Forward search
        # ---------------------------------------------------------------
        start = -d + fs
        end = d - fe

        for k in range(start, end + 1, 2):
            idx = offset + k

            if k == -d or (k != d and vf[idx - 1] < vf[idx + 1]):
                # Insertion: move down.
                x = vf[idx + 1]
            else:
                # Deletion: move right.
                x = vf[idx - 1] + 1

            y = x - k
            x0 = x
            y0 = y

            # Follow the snake.
            while x < n and y < m and A[x] == B[y]:
                x += 1
                y += 1

            vf[idx] = x

            # Remove diagonals that have left the edit graph.
            if x > n:
                fe += 2
                continue

            if y > m:
                fs += 2
                continue

            # For odd delta, the forward and backward searches can overlap.
            if odd:
                kb = delta - k
                if -d < kb < d:
                    xb = vb[offset + kb]
                    if xb != -1 and x + xb >= n:
                        return x0, y0, x, y

        # ---------------------------------------------------------------
        # Backward search
        # ---------------------------------------------------------------
        start = -d + bs
        end = d - be

        for k in range(start, end + 1, 2):
            idx = offset + k

            if k == -d or (k != d and vb[idx - 1] < vb[idx + 1]):
                x = vb[idx + 1]
            else:
                x = vb[idx - 1] + 1

            y = x - k
            x0 = x
            y0 = y

            # Follow the reverse snake.
            while x < n and y < m and Ar[x] == Br[y]:
                x += 1
                y += 1

            vb[idx] = x

            if x > n:
                be += 2
                continue

            if y > m:
                bs += 2
                continue

            # For even delta, check overlap with the forward search.
            if not odd:
                kf = delta - k

                if -d <= kf <= d:
                    xf = vf[offset + kf]
                    if xf != -1 and xf + x >= n:
                        return n - x, m - y, n - x0, m - y0

    raise RuntimeError("middle snake not found")


def diff_marks(a, b):
    """Compute a minimal Myers diff.

    Returns:
        del_a:
            bytearray marking elements deleted from ``a``.

        ins_b:
            bytearray marking elements inserted into ``b``.

    A value of 1 means the element participates in an edit.
    A value of 0 means it is matched.
    """
    na = len(a)
    nb = len(b)

    # Assign a compact integer ID to every distinct item.
    ids = {}
    ia = []
    ib = []

    for item in a:
        value = ids.get(item)
        if value is None:
            value = len(ids)
            ids[item] = value
        ia.append(value)

    for item in b:
        value = ids.get(item)
        if value is None:
            value = len(ids)
            ids[item] = value
        ib.append(value)

    # Items occurring in only one sequence can never be part of a match.
    in_a = set(ia)
    in_b = set(ib)

    ma = [i for i, value in enumerate(ia) if value in in_b]
    mb = [j for j, value in enumerate(ib) if value in in_a]

    fa = [ia[i] for i in ma]
    fb = [ib[j] for j in mb]

    del_f = bytearray(len(fa))
    ins_f = bytearray(len(fb))

    # Each tuple represents:
    #   [a0:a1] -> [b0:b1]
    stack = [(0, len(fa), 0, len(fb))]

    while stack:
        a0, a1, b0, b1 = stack.pop()

        # ---------------------------------------------------------------
        # Strip common prefix.
        # ---------------------------------------------------------------
        while a0 < a1 and b0 < b1 and fa[a0] == fb[b0]:
            a0 += 1
            b0 += 1

        # ---------------------------------------------------------------
        # Strip common suffix.
        # ---------------------------------------------------------------
        while a0 < a1 and b0 < b1 and fa[a1 - 1] == fb[b1 - 1]:
            a1 -= 1
            b1 -= 1

        # Everything on B is an insertion.
        if a0 == a1:
            if b0 < b1:
                ins_f[b0:b1] = b"\x01" * (b1 - b0)
            continue

        # Everything on A is a deletion.
        if b0 == b1:
            del_f[a0:a1] = b"\x01" * (a1 - a0)
            continue

        # ---------------------------------------------------------------
        # Solve the remaining problem using the middle snake.
        # ---------------------------------------------------------------
        A = fa[a0:a1]
        B = fb[b0:b1]

        n = a1 - a0
        m = b1 - b0

        Ar = A[::-1]
        Br = B[::-1]

        # Sentinel values allow the middle-snake implementation to safely
        # perform comparisons at the logical boundary.
        A.append(-1)
        B.append(-2)
        Ar.append(-1)
        Br.append(-2)

        sx, sy, ex, ey = middle_snake(A, B, Ar, Br, n, m)

        # Push right side first so that the left side is processed next.
        stack.append(
            (
                a0 + ex,
                a1,
                b0 + ey,
                b1,
            )
        )
        stack.append(
            (
                a0,
                a0 + sx,
                b0,
                b0 + sy,
            )
        )

    # Start with everything marked as changed. Only elements that survived
    # the filtered Myers search are then replaced with their actual status.
    del_a = bytearray(b"\x01") * na
    ins_b = bytearray(b"\x01") * nb

    for filtered_index, original_index in enumerate(ma):
        del_a[original_index] = del_f[filtered_index]

    for filtered_index, original_index in enumerate(mb):
        ins_b[original_index] = ins_f[filtered_index]

    return del_a, ins_b


def ranges(marks):
    """Convert a 0/1 bytearray into ``start-end,...`` ranges.

    Returns ``.`` when no elements are marked.
    """
    n = len(marks)
    parts = []

    start = marks.find(1)

    while start != -1:
        end = marks.find(0, start)

        if end == -1:
            end = n

        parts.append(f"{start}-{end}")

        if end >= n:
            break

        start = marks.find(1, end)

    return ",".join(parts) if parts else "."


def build_output(a, b, del_a, ins_b, highlight):
    """Build the textual diff output.

    Deletes are emitted before inserts, matching the original behavior.
    """
    na = len(a)
    nb = len(b)

    out = []
    i = j = 0

    while True:
        next_delete = del_a.find(1, i)
        next_insert = ins_b.find(1, j)

        if next_delete == -1 and next_insert == -1:
            break

        # Number of unchanged lines before the next edit.
        count = na - i

        if next_delete != -1:
            count = min(count, next_delete - i)

        if next_insert != -1:
            count = min(count, next_insert - j)

        if count:
            out.extend(
                b" " + line
                for line in a[i:i + count]
            )
            i += count
            j += count

        # ---------------------------------------------------------------
        # Consume one contiguous edit block.
        # ---------------------------------------------------------------
        delete_end = del_a.find(0, i)
        if delete_end == -1:
            delete_end = na

        insert_end = ins_b.find(0, j)
        if insert_end == -1:
            insert_end = nb

        deleted = a[i:delete_end]
        inserted = b[j:insert_end]

        out.extend(b"-" + line for line in deleted)

        if not highlight:
            out.extend(b"+" + line for line in inserted)
        else:
            paired = min(len(deleted), len(inserted))

            for index, line in enumerate(inserted):
                out.append(b"+" + line)

                if index < paired:
                    old = deleted[index].decode(
                        "utf-8",
                        "surrogateescape",
                    )
                    new = line.decode(
                        "utf-8",
                        "surrogateescape",
                    )

                    deleted_marks, inserted_marks = diff_marks(old, new)

                    marker = (
                        f"? {ranges(deleted_marks)} | "
                        f"{ranges(inserted_marks)}"
                    ).encode("utf-8")

                    out.append(marker)

        i = delete_end
        j = insert_end

    # Remaining unchanged tail.
    if i < na:
        out.extend(b" " + line for line in a[i:])

    return out


def main():
    """CLI entry point."""
    if len(sys.argv) != 4:
        print(
            "usage: main.py lines|highlight A_PATH B_PATH",
            file=sys.stderr,
        )
        return 2

    command = sys.argv[1]

    if command not in ("lines", "highlight"):
        print(
            "usage: main.py lines|highlight A_PATH B_PATH",
            file=sys.stderr,
        )
        return 2

    try:
        a = read_lines(sys.argv[2])
        b = read_lines(sys.argv[3])
    except OSError as exc:
        print(
            f"error: cannot read file: {exc}",
            file=sys.stderr,
        )
        return 2

    del_a, ins_b = diff_marks(a, b)

    output = build_output(
        a,
        b,
        del_a,
        ins_b,
        command == "highlight",
    )

    if output:
        sys.stdout.buffer.write(b"\n".join(output) + b"\n")
        sys.stdout.buffer.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())