"""Scenario evaluations: a simulated caller (an LLM) talks to the real agent, and the outcomes
are graded by code. Calls real LLMs and costs money, so it is deliberately NOT part of pytest:

    .venv/bin/python -m evals --help

The harness itself (checks, caller plumbing, reporting) is covered by offline tests in
tests/test_evals*.py.
"""
