#!/usr/bin/env python
"""Write a single-file copy of a UEL source with its module includes pasted in.

    python scripts/make_flat_uel.py test/uel_nlmech_pk1.for work/uel_flat/uel_flat.for

The UEL sources contain lines such as ``include '../module/linear_algebra.for'``, which only
resolve when Abaqus compiles them from inside ``test/``. The flat copy can be used from any
job directory with ``user=<path>/uel_flat.for`` (config: ``[abaqus] user_source``), when no
precompiled library is available.

The original files are not modified. ``INCLUDE 'ABA_PARAM.INC'`` lines are left as they
are: Abaqus provides that file. Bytes that are not ASCII (accents in comments) are copied
unchanged. Standard library only.
"""

import os
import re
import sys

INCLUDE = re.compile(r"^\s+include\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)


def flatten(path, seen=None):
    seen = seen if seen is not None else []
    path = os.path.abspath(path)
    if path in seen:
        raise RuntimeError("circular include: %s" % path)
    out = []
    with open(path, "r", encoding="latin-1", newline="") as fh:  # latin-1 keeps every byte
        for line in fh:
            m = INCLUDE.match(line)
            if m and m.group(1).upper() != "ABA_PARAM.INC":
                target = os.path.join(os.path.dirname(path), m.group(1))
                if not os.path.isfile(target):
                    raise RuntimeError("%s includes %s, which does not exist" % (path, target))
                out.append("!     >>> begin include %s\n" % m.group(1))
                out.extend(flatten(target, seen + [path]))
                out.append("!     <<< end include %s\n" % m.group(1))
            else:
                out.append(line.rstrip("\r\n") + "\n")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dst = argv
    lines = flatten(src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with open(dst, "w", encoding="latin-1", newline="\n") as fh:
        fh.writelines(lines)
    print("wrote %s (%d lines)" % (dst, len(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
