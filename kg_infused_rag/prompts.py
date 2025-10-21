INSTRUCTION_DICT = {
    "base_wo_retri": "Only give me the answer and do not output any other words.",
    "base_retri": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "base_retri_with_facts": "Answer the question based on the given passages and facts. Only give me the answer and do not output any other words."
    }


PROMPT_DICT = {
    # Question Answering 
    "base_wo_retri": (
        "{instruction_wo_retri}\n\n"
        "Question: {question}\n"
        "Answer: "
    ),
    "base_retri": (
        "{instruction_retri}\n\n"
        "Passages:\n{passages}\n\n"
        "Question: {question}\n"
        "Answer: "
    ),


    # 1. Note/Summary
    ## 1.1 write a note of retrieved passages
    "write_note": (
        "Based on the provided document content, write a note. The note should integrate all relevant information from the original text that can help answer the specified question and form a coherent paragraph. Please ensure that the note includes all original text information useful for answering the question.\n\n"
        "Question to be answered:\n{question}\n\n"
        "Document content:\n{passages}\n\n"
        "Note:\n"
    ), 
    ## 1.2 write a summary of retrieved triples
    "triples_summary": (
        "Given a question and a set of retrieved entity triples, write a summary that captures the key information from the triples. "
        "If the triples do not provide enough information to directly answer the question, still summarize the information provided in the triples, even if it does not directly relate to the question. "
        "Focus on presenting all available details, regardless of their direct relevance to the query, in a concise and informative way.\n\n"
        "Question:\n{question}\n\n"
        "Selected Triples:\n{selected_triples}\n\n"
        "Summary:\n"
    ), 


    # 2. Augment Passages
    "aug_passage": (
        "You are an expert in text enhancement and fact integration. Given a question, a retrieved passage, and relevant factual information, your task is to improve the passage by seamlessly incorporating useful details from the factual information. Ensure that the enhanced passage remains coherent, well-structured, and directly relevant to answering the question. Preserve the original meaning while making the passage more informative. Avoid introducing unrelated content.\n\n"
        "Question:\n{question}\n\n"
        "Retrieved Passage:\n{passages}\n\n"
        "Relevant Factual Information:\n{facts}\n\n"
        "Enhanced passage:\n"
    ),


    # 3. Query Expansion
    ## 3.1 vanilla QE
    "query_expansion_only_query": (
        "Generate a new short query that is distinct from but closely related to the original question. This new query should aim to retrieve additional passages that fill in gaps or provide complementary knowledge necessary to thoroughly address the original question. Ensure the new query is relevant, precise, and broadens the scope of information tied to the original question. Only give me the new short query and do not output any other words.\n\n"
        "Original Question:\n{question}\n\n"
        "New Query:\n"
    ), 
    ## 3.2 QE based on summary of triples 
    "query_expansion_query_and_triples_summary": (
        "Generate a new short query that is distinct from but closely related to the original question. This new query should leverage both the original question and the provided paragraph to retrieve additional passages that fill in gaps or provide complementary knowledge necessary to thoroughly address the original question. Ensure the new query is relevant, precise, and broadens the scope of information tied to the original question. Only give me the new short query and do not output any other words.\n\n"
        "Original Question:\n{question}\n\n"
        "Related Paragraph:\n{triples_summary}\n\n"
        "New Query:\n"
    ),


    # 4. Triples Selection and Update
    "triples_selection_before_retri": (
        "Given a question and a set of retrieved entity triples, select only the triples that are relevant to the question.\n\n"
        "Information:\n"
        "1. Each triple is in the form of <subject, predicate, object>.\n"
        "2. The objects in the selected triples will be further explored in the next steps to gather additional relevant triples information.\n\n"
        "Rules:\n"
        "1. Only select triples from the retrieved set. Do not generate new triples.\n"
        "2. A triple is relevant if it contains information about entities or relationships that are important for answering the question, either directly or indirectly.\n"
        "   - For example, if the question asks about a specific person, include triples about that person's name, occupation, relationships, etc.\n"
        "   - If the question asks about an event or entity, include related background information that can help answer the question.\n"
        "3. Output triples exactly as they appear in angle brackets (<...>).\n\n"
        "Question:\n{question}\n\n"
        "Retrieved Entity Triples:\n{triples}\n\n"
        "Selected Triples:\n"
    ),
    "triples_update_before_retri": (
        "Given a question, a set of previously selected entity triples that are relevant to the question, and a new set of retrieved entity triples, "
        "select only the triples from the new set of retrieved entity triples that expand or enhance the information provided by the previously selected triples to help address the question.\n\n"
        "Information:\n"
        "1. Each triple is in the form of <subject, predicate, object>.\n"
        "2. The objects in the selected triples will be further explored in the next steps to gather additional relevant triples information.\n\n"
        "Rules:\n"
        "1. Only select triples from the new set of retrieved entity triples. Do not include duplicates of the previously selected triples or generate new triples.\n"
        "2. A triple is considered relevant if it:\n"
        "   - Provides new information that complements or builds upon the entities, relationships, or concepts in the previously selected triples, and\n"
        "   - Helps to better address or provide context for answering the question.\n"
        "3. Do not include triples that are unrelated to the question or do not expand on the previously selected triples.\n"
        "4. Output triples exactly as they appear in angle brackets (<...>).\n\n"
        "Question:\n{question}\n\n"
        "Previously Selected Triples:\n{previous_selected_triples}\n\n"
        "New Retrieved Entity Triples:\n{new_retrieved_triples}\n\n"
        "Selected Triples:\n"
    ),


    # Prompts of Self-RAG
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    )
}


def prompts_setup(args):
    """
    Setup all prompts for triples exploration, query expansion, passage augmentation, and answer generation.
    Returns:
        dict: All prompts keyed by usage (same keys as original pipeline expects).
    """
    prompt_dict = {}

    # --- 1) Base prompts from config ---
    base_prompts = {
        "triples_selection": PROMPT_DICT[args.prompt_name_of_triples_selection],
        "triples_update": PROMPT_DICT[args.prompt_name_of_triples_update],
        "triples_summary": PROMPT_DICT[args.prompt_name_of_triples_summary],
        "triples_based_qe": PROMPT_DICT[args.prompt_name_of_triples_based_qe],
        "aug_passage": PROMPT_DICT[args.prompt_aug_passage],
        "wo_retri": PROMPT_DICT[args.prompt_wo_retri],
        "retri_raw": PROMPT_DICT[args.prompt_retri],
    }
    # passages_note prompts only if enabled (match original behavior)
    if getattr(args, "passages_note", False):
        base_prompts["passages_note"] = PROMPT_DICT[args.prompt_passages_note]
        base_prompts["retri_raw"] = base_prompts["retri_raw"].replace("{passages}", "{passages_note}")
    else:
        base_prompts["passages_note"] = None  # do not create dependent prompts later


    # --- 2) Static prompts outside rounds ---
    prompt_dict["wo_retri"] = base_prompts["wo_retri"]
    prompt_dict["aug_passage"] = base_prompts["aug_passage"]

    if base_prompts["passages_note"]:
        # when passages_note==True, retri_raw uses {passages_note}
        prompt_retri_raw = base_prompts["retri_raw"].replace("{passages}", "{passages_note}")
        prompt_dict["retri_raw"] = prompt_retri_raw
        # retri_vanilla_qe also uses {passages_note}
        prompt_dict["retri_vanilla_qe"] = prompt_retri_raw.replace("{passages_note}", "{passages_note_of_vanilla_qe}")
        # also expose passages_note prompts (vanilla_qe note depends on it)
        prompt_dict["passages_note"] = base_prompts["passages_note"]
        prompt_dict["passages_note_vanilla_qe"] = base_prompts["passages_note"].replace("{passages}", "{passages_of_vanilla_qe}")
    else:
        # when passages_note==False, retri_raw uses {passages}
        prompt_dict["retri_raw"] = base_prompts["retri_raw"]
        prompt_dict["retri_vanilla_qe"] = base_prompts["retri_raw"].replace("{passages}", "{passages_of_vanilla_qe}")


    # --- 3) Triples selection & update / summaries / QE per round ---
    prompt_dict["triples_selection_of_round_0"] = base_prompts["triples_selection"].replace(
        "{triples}", "{triples_text_of_round_0}"
    )
    for round_id in range(args.N_MAX_EXPLORE_ROUNDS):
        pre_sel = f"{{selected_triples_text_of_round_{round_id-1}}}"
        cur_triples = f"{{triples_text_of_round_{round_id}}}"
        cur_new_triples = f"{{new_triples_text_of_round_{round_id}}}"
        cur_sel = f"{{selected_triples_text_of_round_{round_id}}}"

        # triples update (from round 1+)
        if round_id >= 1:
            prompt_dict[f"triples_update_of_round_{round_id}"] = (
                base_prompts["triples_update"]
                .replace("{previous_selected_triples}", pre_sel)
                .replace("{new_retrieved_triples}", cur_new_triples)
            )

        # triples summary (before/after)
        prompt_dict[f"triples_summary_before_selection_of_round_{round_id}"] = \
            base_prompts["triples_summary"].replace("{selected_triples}", cur_triples)
        prompt_dict[f"triples_summary_after_selection_of_round_{round_id}"] = \
            base_prompts["triples_summary"].replace("{selected_triples}", cur_sel)

        # triples-based QE (before/after)
        prompt_dict[f"triples_based_qe_before_selection_of_round_{round_id}"] = \
            base_prompts["triples_based_qe"].replace(
                "{triples_summary}",
                f"{{triples_summary_before_selection_of_round_{round_id}}}"
            )
        prompt_dict[f"triples_based_qe_after_selection_of_round_{round_id}"] = \
            base_prompts["triples_based_qe"].replace(
                "{triples_summary}",
                f"{{triples_summary_after_selection_of_round_{round_id}}}"
            )

        # passages note and augmented passages (only if passages_note is enabled)
        if base_prompts["passages_note"]:
            # passages note for triples-based QE merged passages
            prompt_dict[f"passages_note_triples_based_qe_before_selection_of_round_{round_id}"] = \
                base_prompts["passages_note"].replace(
                    "{passages}",
                    f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}"
                )
            prompt_dict[f"passages_note_triples_based_qe_after_selection_of_round_{round_id}"] = \
                base_prompts["passages_note"].replace(
                    "{passages}",
                    f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}"
                )

            # augmented passages with triples summary (raw passages or QE passages -> into a note)
            # 1) raw passages + triples summary note
            prompt_dict[f"aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", "{passages_note}"
                ).replace(
                    "{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}"
                )
            prompt_dict[f"aug_passage_with_triples_summary_after_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", "{passages_note}"
                ).replace(
                    "{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}"
                )

            # 2) triples-based QE passages + triples summary note
            prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", f"{{passages_note_of_triples_based_qe_before_selection_of_round_{round_id}}}"
                ).replace(
                    "{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}"
                )
            prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", f"{{passages_note_of_triples_based_qe_after_selection_of_round_{round_id}}}"
                ).replace(
                    "{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}"
                )

            # RAG generation prompts (match original branching)
            if args.kg_based_qe:
                prompt_dict[f"rag_with_triples_based_qe_passages_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_of_triples_based_qe_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_triples_based_qe_passages_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_of_triples_based_qe_after_selection_of_round_{round_id}}}"
                    )

            if args.kg_aug_ag:
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_aug_triples_summary_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_aug_triples_summary_after_selection_of_round_{round_id}}}"
                    )

            if args.kg_based_qe and args.kg_aug_ag:
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_of_triples_based_qe_aug_triples_summary_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages_note}",
                        f"{{passages_note_of_triples_based_qe_aug_triples_summary_after_selection_of_round_{round_id}}}"
                    )

        else:
            # augmented passages with triples summary (raw passages or QE passages -> into a note)
            # 1) raw passages + triples summary note
            prompt_dict[f"aug_passage_with_triples_summary_before_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}"
                )
            prompt_dict[f"aug_passage_with_triples_summary_after_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}"
                )
            # 2) triples-based QE passages + triples summary note
            prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_before_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}"
                ).replace(
                    "{facts}", f"{{triples_summary_before_selection_of_round_{round_id}}}"
                )
            prompt_dict[f"aug_triples_based_qe_passage_with_triples_summary_after_selection_of_round_{round_id}"] = \
                base_prompts["aug_passage"].replace(
                    "{passages}", f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}"
                ).replace(
                    "{facts}", f"{{triples_summary_after_selection_of_round_{round_id}}}"
                )

            # RAG generation prompts (match original branching)
            if args.kg_based_qe:
                prompt_dict[f"rag_with_triples_based_qe_passages_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_of_triples_based_qe_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_triples_based_qe_passages_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_of_triples_based_qe_after_selection_of_round_{round_id}}}"
                    )

            if args.kg_aug_ag:
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_aug_triples_summary_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_raw_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_aug_triples_summary_after_selection_of_round_{round_id}}}"
                    )

            if args.kg_based_qe and args.kg_aug_ag:
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_before_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_of_triples_based_qe_aug_triples_summary_before_selection_of_round_{round_id}}}"
                    )
                prompt_dict[f"rag_with_triples_based_qe_passages_aug_triples_summary_after_selection_of_round_{round_id}"] = \
                    base_prompts["retri_raw"].replace(
                        "{passages}",
                        f"{{passages_of_triples_based_qe_aug_triples_summary_after_selection_of_round_{round_id}}}"
                    )

    return prompt_dict
