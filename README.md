# kepy

## Description

Word-based keyphrase extraction module in Python.

A typical usage of this module is:

    import kepy

    # Text file from which keyphrases are extracted. Input file is expected to 
    # be in one tokenized, POS tagged sentence per line format.
    input_file = "/path/to/input.txt"

    # create the extractor
    e = kepy.WordBasedILPKeyphraseExtractor(args.input, use_stems=True)

    # load the document
    e.read_document()

    # candidate selection
    e.select_candidates()

    # compute the scores using random walk
    scores = e.random_walk_word_scoring()

    # rank candidates
    candidates = e.rank_candidates_with_sum(scores)

    # print the 10 best keyphrases
    keyphrases = ';'.join([' '.join(u[1]) for u in candidates[:10]])
