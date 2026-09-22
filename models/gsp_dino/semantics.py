"""Anatomy semantics selected by the active dataset profile."""
from datasets.specs import active_dataset_spec, relation_matrix


_SPEC = active_dataset_spec()
ANATOMY_PROMPTS = list(_SPEC.anatomy_prompts)
ORGAN_NAMES = list(_SPEC.organ_names)
ANATOMY_RELATION_EDGES = list(_SPEC.anatomy_edges)


def get_anatomy_relation_matrix(num_classes: int = 14, self_loop: bool = True):
    return relation_matrix(_SPEC, num_classes, self_loop)
