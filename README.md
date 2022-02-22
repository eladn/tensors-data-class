# TensorsDataClass
PyTorch Extension Library for organizing tensors in a form of a structured tree of dataclasses, with built-in support for advanced collating mechanisms. The batch creation process seamlessly solves issues like: sequences padding, un/flattening variable #objects per example into a single batch dimension, fixing within-example indices to be batch-based indices, auto-creation of sequences & collate masks, and more.

# What pains TensorsDataClass aims to solve
Sometimes the input data to the model is formed of several types of elements; that is, each pre-processed example contains multiple kinds of entities. For example, a model for programming-related tasks that uses the code's underlying structure to represent the input code-snippet might have the following elements: CFG nodes, AST nodes, tokens, symbols, identifiers, and paths. Some of the model's calculations sometimes involves elements of multiple kinds, and an element can participate in multiple calculations. Typically, the index of an element can be used to address it in the computation. For example, one could use `all_identifiers_encodings[identifier_indices_occur_in_code_statement]` to get the encodings of certain identifiers. Another example is sequences of elements, that can be given as sequences of indices `sequence_indices`. After encoding the elements `all_elements_encodings`, the concrete seqeunce of encoded elements can be addressed by `all_elements_encodings[sequence_indices]`.

Usually, training and evaluating neural networks is performed over batches of examples, following the SIMD (single instruction multiple data) computational scheme to maximize the utility of the accelerated processing units and make the training feasible under the available resources. However, the preprocessed example is stored on its own, while it should reoccur in various batches during training. Therefore, the batching takes place during data loading.

Whenever a collection of examples are being collated into a batch, continuous tensors are being created containing all the elements in the batch. As a result, the indices of these elements are updated. Thus, the references to them have to be fixed accordingly to retain the indexing consistency.

... variable number of sequences per example where the sequence lengths may also be variable; lots of inputs usually gets messy - hard to handle, to name, to move to GPU, to abstract in a (X,Y) fashion ...

# Installation
```bash
pip install tensors-data-class
```

# Usage example
```python
# TODO: simplify the below example. still use these:
#   BatchFlattenedSeq, BatchFlattenedTensor,
#   BatchedFlattenedIndicesTensor,
#   BatchedFlattenedIndicesFlattenedTensor,
#   BatchedFlattenedIndicesFlattenedSeq,
#   BatchedFlattenedIndicesPseudoRandomPermutationBatchedFlattenedIndicesPseudoRandomPermutation,
#   BatchFlattenedPseudoRandomSamplerFromRange

from tensors_data_class import *

@dataclasses.dataclass
class CodeExpressionTokensSequenceInputTensors(TensorsDataClass):
    token_type: BatchFlattenedSeq  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    kos_token_index: BatchFlattenedTensor  # (nr_kos_tokens_in_all_expressions_in_batch,)
    identifier_index: BatchedFlattenedIndicesFlattenedTensor  # (nr_identifier_tokens_in_all_expressions_in_batch,)


@dataclasses.dataclass
class SymbolsInputTensors(TensorsDataClass):
    symbols_identifier_indices: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_in_batch,);  value meaning: identifier batched index
    symbols_appearances_symbol_idx: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_appearances,);
    symbols_appearances_expression_token_idx: BatchFlattenedTensor = None  # (nr_symbols_appearances,);
    symbols_appearances_cfg_expression_idx: BatchedFlattenedIndicesFlattenedTensor = None  # (nr_symbols_appearances,);


@dataclasses.dataclass
class CFGPathsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq
    edges_types: BatchFlattenedSeq


@dataclasses.dataclass
class CFGPathsNGramsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq
    edges_types: BatchFlattenedSeq


@dataclasses.dataclass
class PDGInputTensors(TensorsDataClass):
    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_has_expression_mask: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_tokenized_expressions: Optional[CodeExpressionTokensSequenceInputTensors] = None
    cfg_nodes_random_permutation: Optional[BatchedFlattenedIndicesPseudoRandomPermutation] = None
    cfg_control_flow_paths: Optional[CFGPathsInputTensors] = None
    cfg_control_flow_paths_ngrams: Optional[Dict[int, CFGPathsNGramsInputTensors]] = None


@dataclasses.dataclass
class IdentifiersInputTensors(TensorsDataClass):
    sub_parts_batch: BatchFlattenedTensor  # (nr_sub_parts_in_batch, )
    identifier_sub_parts_index: BatchedFlattenedIndicesFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_vocab_word_index: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_hashings: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier, nr_hashing_features)
    sub_parts_obfuscation: BatchFlattenedPseudoRandomSamplerFromRange  # (nr_sub_parts_obfuscation_embeddings)


@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    example_hash: str
    identifiers: IdentifiersInputTensors
    symbols: SymbolsInputTensors
    method_tokenized_code: Optional[CodeExpressionTokensSequenceInputTensors] = None
    pdg: Optional[PDGInputTensors] = None

example1 = MethodCodeInputTensors(...)  # TODO: fill example data
example2 = MethodCodeInputTensors(...)  # TODO: fill example data
batch = MethodCodeInputTensors.collate([example1, example2])
print(batch)
# TODO: add example for creating a padded-sequence (after applying embedding on the input), unflattening. 
```

# Different types for different use-cases
## TensorsDataClass
## BatchFlattenedTensor
## BatchFlattenedSeq
## BatchedFlattenedIndicesTensor
## BatchedFlattenedIndicesFlattenedTensor
## BatchedFlattenedIndicesFlattenedSeq
## BatchedFlattenedIndicesPseudoRandomPermutationBatchedFlattenedIndicesPseudoRandomPermutation
## BatchFlattenedPseudoRandomSamplerFromRange
