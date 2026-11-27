"""
src.dask_matrix.reduction — Binary-Tree Reduction Graph Builder

Constructs the binary-tree reduction for summing the Nb partial products P_ijk
into the final output block C_ij, adhering strictly to SPEC.md:
- Round-based binary tree: reduces m inputs to ceil(m / 2) intermediate futures.
- Depth: ceil(log2(Nb)) reduction stages.
- Additions per output block: exactly Nb - 1 addition tasks.
- Unique task keys per repetition to prevent silent Dask result deduplication/caching.
"""

from typing import Any, Callable, List, Optional
from src.dask_matrix.kernels import block_add


def build_binary_tree_reduction(
    client: Any,
    partial_product_futures: List[Any],
    block_row: int,
    block_col: int,
    repetition_id: int = 0,
    add_fn: Optional[Callable] = None,
    key_namespace: Optional[str] = None,
) -> Any:
    """
    Builds a binary-tree reduction graph across a list of Nb partial-product futures.

    Args:
        client: Dask distributed Client instance.
        partial_product_futures: List of Nb futures representing P_ijk for k in 0..Nb-1.
        block_row: Row index i of the output block.
        block_col: Column index j of the output block.
        repetition_id: Unique repetition counter for key uniqueness.
        add_fn: Callable for addition; defaults to block_add.
        key_namespace: Optional invocation/dimension namespace to prevent key collisions
            across repeated calls on a persistent Client.

    Returns:
        A single Dask future representing the fully accumulated block C_ij.

    Raises:
        ValueError: If partial_product_futures is empty.
    """
    if not partial_product_futures:
        raise ValueError(
            f"Cannot reduce empty partial product list for block ({block_row}, {block_col})."
        )

    if len(partial_product_futures) == 1:
        return partial_product_futures[0]

    current_layer = list(partial_product_futures)
    addition_fn = add_fn or block_add
    round_idx = 0
    ns_prefix = f"add-{key_namespace}-" if key_namespace else "add-"

    while len(current_layer) > 1:
        next_layer = []
        num_items = len(current_layer)

        for idx in range(0, num_items, 2):
            if idx + 1 < num_items:
                # Pair of futures to reduce
                left = current_layer[idx]
                right = current_layer[idx + 1]
                # Enforce unique task key per namespace, repetition, block coordinates, round, and pair
                task_key = (
                    f"{ns_prefix}r{repetition_id}-b({block_row},{block_col})"
                    f"-round{round_idx}-pair{idx // 2}"
                )
                sum_future = client.submit(
                    addition_fn,
                    left,
                    right,
                    key=task_key,
                    pure=False,  # Avoid global cache collisions across runs
                )
                next_layer.append(sum_future)
            else:
                # Odd item carries forward untouched to next reduction stage
                next_layer.append(current_layer[idx])

        current_layer = next_layer
        round_idx += 1

    return current_layer[0]
