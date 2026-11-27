"""Append the MTP layer to a copy of a q4_0_q6 weight file of Qwen3.6-35B-A3B (scratch tool).

    append_mtp.py SCRIPTS_DIR MODEL_DIR FILE

FILE must be a clone of the weight file without the MTP layer. The script checks that the data
start stays where it is, writes the MTP tensors after the last tensor, and then rewrites the
header and tensor table as write_weight_file would for the config with mtp_layers = 1.
"""
import struct
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import convert  # noqa: E402
import qwen_weights as q  # noqa: E402

model_dir, path = sys.argv[2], sys.argv[3]
old_config, old = q.read_weight_file(path)
assert old_config["weight_format"] == q.WEIGHT_FORMATS["q4_0_q6"] and old_config.get("mtp_layers", 0) == 0
hf_config, hf = q.config_from_hugging_face(model_dir), q.read_safetensors(model_dir)
assert hf_config["mtp_layers"] == 1
config, tensors = convert.convert_tensors(hf_config, hf)
config, tensors = q.quantize(config, tensors, "q4_0_q6")
for field in q.CONFIG_INTS:
    if field != "mtp_layers":
        assert int(config[field]) == int(old_config[field]), field
names, shapes = q.tensor_names(config), q.tensor_shapes(config)
old_names = q.tensor_names(old_config)
assert names[: len(old_names)] == old_names == list(old)

with open(path, "rb") as f:
    head = bytearray(f.read(q.TABLE_OFFSET + q.ENTRY_BYTES * len(names)))
old_start = -(-(q.TABLE_OFFSET + q.ENTRY_BYTES * len(old_names)) // q.ALIGNMENT) * q.ALIGNMENT
data_start = -(-(q.TABLE_OFFSET + q.ENTRY_BYTES * len(names)) // q.ALIGNMENT) * q.ALIGNMENT
assert data_start == old_start, "the longer table would move the data"
last = q.TABLE_OFFSET + (len(old_names) - 1) * q.ENTRY_BYTES
offset, nbytes = struct.unpack_from("<2q", head, last + 64 + 48)
offset += -(-nbytes // q.ALIGNMENT) * q.ALIGNMENT

entries = []
with open(path, "r+b") as out:
    out.seek(0, 2)
    assert out.tell() == offset, (out.tell(), offset)
    for name in names[len(old_names):]:
        dtype, array = q.resolve(tensors[name])
        shape = shapes[name]
        assert tuple(array.shape) == q.storage_shape(dtype, shape), name
        assert array.dtype == q.STORAGE[dtype], name
        size = int(np.prod(shape)) * q.DTYPE_BITS[dtype] // 8
        dims = list(shape) + [0] * (4 - len(shape))
        entry = name.encode("ascii").ljust(64, b"\0")
        assert len(entry) == 64, name
        entry += struct.pack("<8q", q.DTYPE_CODES[dtype], len(shape), *dims, offset, size)
        entries.append(entry)
        out.seek(offset)
        out.write(np.ascontiguousarray(array).data)
        offset += -(-size // q.ALIGNMENT) * q.ALIGNMENT
        print(name, dtype, shape, flush=True)
    out.truncate(offset)
    # The header: the count of tensors and the config integers, as write_weight_file writes them.
    version = q.file_version(config)
    words = [version, len(names), q.ALIGNMENT, q.TABLE_OFFSET] + [int(config[f]) for f in q.CONFIG_INTS[: q.CONFIG_INT_SLOTS[version]]]
    words += [0] * (q.CONFIG_INT_SLOTS[version] - len(q.CONFIG_INTS[: q.CONFIG_INT_SLOTS[version]]))
    struct.pack_into(f"<{len(words)}q", head, 8, *words)
    for i, entry in enumerate(entries):
        at = q.TABLE_OFFSET + (len(old_names) + i) * q.ENTRY_BYTES
        head[at : at + q.ENTRY_BYTES] = entry
    out.seek(0)
    out.write(head)
print("wrote", len(names), "tensors, end", offset)
