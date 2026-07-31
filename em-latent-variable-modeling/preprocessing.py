from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().replace('&', '&amp;')
    root = ET.fromstring(text)

    def parse_pairs(node):
        if node is None or node.text is None:
            return []
        return [tuple(map(int, pair.split('-'))) for pair in node.text.split()]

    sentence_pairs, alignments = [], []
    for s in root.findall('s'):
        source = s.find('english').text.split()
        target = s.find('czech').text.split()
        sentence_pairs.append(SentencePair(source, target))
        alignments.append(LabeledAlignment(
            sure=parse_pairs(s.find('sure')),
            possible=parse_pairs(s.find('possible')),
        ))
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    src_counter, tgt_counter = Counter(), Counter()
    for pair in sentence_pairs:
        src_counter.update(pair.source)
        tgt_counter.update(pair.target)

    def build(counter):
        
        return {tok: i for i, (tok, _) in enumerate(counter.most_common(freq_cutoff))}

    return build(src_counter), build(tgt_counter)


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        src = [source_dict[t] for t in pair.source if t in source_dict]
        tgt = [target_dict[t] for t in pair.target if t in target_dict]
        if src and tgt:                          
            tokenized_sentence_pairs.append(TokenizedSentencePair(
                np.array(src, dtype=np.int32),
                np.array(tgt, dtype=np.int32),
            ))
    return tokenized_sentence_pairs
