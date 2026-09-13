"""Discourse edges only; model-generated readout edges are added separately."""


def build_discourse_edges(num_nodes, parsed_edges, variant, special_relation=16):
    """Node 0 is the dummy readout node; other nodes are utterances."""
    parsed_map = {(head, tail): relation for head, tail, relation in parsed_edges if head != 0}
    edges, types, priors = [], [], []
    if variant == 'original_hard':
        for head, tail, relation in parsed_edges:
            if head != 0:
                edges.append([head, tail])
                types.append(relation)
                priors.append(1.0)
    elif variant == 'learnable_soft':
        for tail in range(1, num_nodes):
            for head in range(1, num_nodes):
                if head == tail:
                    continue
                edges.append([head, tail])
                types.append(parsed_map.get((head, tail), special_relation))
                priors.append(1.0 if (head, tail) in parsed_map else 0.5)
    else:
        raise ValueError(f'Unsupported expert fusion mode: {variant}')
    return edges, types, priors
