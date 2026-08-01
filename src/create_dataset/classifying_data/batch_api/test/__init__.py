"""OpenAI Batch API labeling for the relation-classification dataset.

Wraps the validated synchronous labeler (``gpt-5.2`` @ ``medium`` reasoning with the
``short2b`` prompt — see ``estimate_frontier_cost.py``) in OpenAI's Batch API so the
same request is run at the 50%-off bulk rate. See ``label_batch_openai.py``.
"""
