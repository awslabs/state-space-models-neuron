import neuronxcc.nki.language as nl

def index_view(tensor_view, *indexes):
    tensor = tensor_view.base
    new_indexes = []
    for index, offset, scale in zip(indexes, tensor_view._offsets, tensor_view._scales):
        # if isinstance(index, slice):
        #     begin = index.start * scale + offset if index.start is not None else offset
        #     end = index.end * scale + offset if index.start is not None else offset +
        #     step = index.step * scale if index.step is not None else None
        #     begin, end, step = index.start, index.stop, index.step
        #     new_index = slice(index.start * scale + offset, index.stop * scale + offset, index.step * scale)
        # else:
        new_index = index * scale + offset
        new_indexes.append(new_index)
    return tensor.__getitem__(tuple(new_indexes))

def chunk(index, size):
    return nl.ds(index * size, size)