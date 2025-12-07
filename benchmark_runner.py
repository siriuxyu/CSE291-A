#!/usr/bin/env python
"""
Benchmark runner for the CSE291-A agent.

This script knows how to run two kinds of benchmarks:

1. Tool-usage benchmarks (short / medium / long):
   - JSON schema: {"test_cases": [{"id", "type", "expected_tools", "conversation": [...] }]}
   - Each test case has a single user message and an "agent_expected" message.
   - We send the user message to the agent via the /invoke HTTP API and
     check:
       * whether the agent used the expected tools, and
       * whether the final response roughly matches the expected text.

2. LoCoMo-style long-term memory benchmark (locomo1_converted.json):
   - JSON schema: {"test_cases": [...conversation sessions...], "qa": [...] }
   - test_cases contain conversations with "dia_id" fields.
   - qa contains questions with:
         { "question", "answer"?, "evidence": [dia_ids...], "category", "adversarial_answer"? }
   - For each QA item we build a short context from the referenced
     evidence turns, ask the question, and compare the model's answer.

The agent is accessed over HTTP using the FastAPI server from this repo.
Make sure the server is running, e.g.:

    uvicorn server:app --reload

Example usage:

    python benchmark_runner.py --dataset short -o short_results.json
    python benchmark_runner.py --dataset locomo

You can override server URL, model, etc. via CLI flags or environment:

    BENCHMARK_SERVER_URL
    MODEL
    SYSTEM_PROMPT
    MAX_SEARCH_RESULTS
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import requests
import re

from agent.interfaces import SessionMetadata


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return it as a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    """Lower-case + whitespace-normalized representation for fuzzy matching."""
    if value is None:
        return ""
    s = str(value)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_tools_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Collect tool names from a list of messages returned by the agent."""
    tools: List[str] = []
    for m in messages or []:
        tool_calls = m.get("tool_calls") or []
        for call in tool_calls:
            name = call.get("name")
            if name:
                tools.append(name)
    return tools


def pair_conversation_turns(
    conversation: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """
    Pair user messages with their subsequent expected agent messages (if any).

    Returns a list of (user_message, expected_reply) tuples. Some tuples may not
    have an expected reply if the benchmark data omits it.
    """
    pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    pending_user: Optional[Dict[str, Any]] = None

    for turn in conversation:
        role = (turn.get("role") or "").lower()
        if role == "user":
            if pending_user:
                pairs.append((pending_user, None))
            pending_user = turn
        elif role == "agent_expected" and pending_user:
            pairs.append((pending_user, turn))
            pending_user = None

    if pending_user:
        pairs.append((pending_user, None))
    return pairs


def create_session_metadata(
    session_id: str,
    user_id: str,
    turn_count: int,
) -> SessionMetadata:
    """Build a minimal SessionMetadata snapshot for reporting."""
    message_count = max(turn_count * 2, 0)
    now = datetime.utcnow()
    return SessionMetadata(
        session_id=session_id,
        user_id=user_id,
        created_at=now,
        last_active=now,
        message_count=message_count,
        is_finalized=False,
    )


def metadata_to_dict(metadata: SessionMetadata) -> Dict[str, Any]:
    """Convert SessionMetadata dataclass into a JSON-serializable dict."""
    data = asdict(metadata)
    data["created_at"] = metadata.created_at.isoformat()
    data["last_active"] = metadata.last_active.isoformat()
    return data


def execute_user_turn(
    user_content: str,
    userid: str,
    *,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
) -> Tuple[str, List[Dict[str, Any]], float]:
    """Send a single user turn to the agent and capture the response."""
    messages_in = [{"role": "user", "content": user_content}]
    start = time.time()
    final_response, messages_out = call_agent(
        messages_in,
        server_url=server_url,
        userid=userid,
        system_prompt=system_prompt,
        model=model,
        max_search_results=max_search_results,
        timeout=timeout,
    )
    latency = time.time() - start
    return final_response, messages_out, latency


def process_conversation(
    conversation: List[Dict[str, Any]],
    *,
    userid: str,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    verbose: bool,
    label: str,
    evaluate_answers: bool = True,
) -> Tuple[
    List[Dict[str, Any]],
    Counter,
    int,
    int,
]:
    """
    Replay a conversation turn-by-turn and evaluate agent responses.

    Returns:
        turn_details: List of per-turn dictionaries
        tool_counter: Counter of observed tools
        answers_expected: Number of expected answers
        answers_correct: Number of answers that matched expectations
    """
    turn_pairs = pair_conversation_turns(conversation)
    turn_details: List[Dict[str, Any]] = []
    tool_counter: Counter = Counter()
    answers_expected = 0
    answers_correct = 0

    for turn_index, (user_msg, expected_msg) in enumerate(turn_pairs, start=1):
        user_content = user_msg.get("content", "")
        final_response, messages_out, latency = execute_user_turn(
            user_content,
            userid,
            server_url=server_url,
            system_prompt=system_prompt,
            model=model,
            max_search_results=max_search_results,
            timeout=timeout,
        )
        tools_this_turn = extract_tools_from_messages(messages_out)
        tool_counter.update(tools_this_turn)
        
        # Count tool calls per tool type for this turn
        tool_call_counts = Counter(tools_this_turn)
        used_tools_unique = sorted(set(tools_this_turn))
        total_tool_calls_this_turn = len(tools_this_turn)

        ans_ok: Optional[bool] = None
        exp_content = expected_msg.get("content") if expected_msg else None
        if evaluate_answers and expected_msg and exp_content:
            answers_expected += 1
            exp_norm = normalize_text(exp_content)
            resp_norm = normalize_text(final_response)
            ans_ok = bool(exp_norm) and exp_norm in resp_norm
            if ans_ok:
                answers_correct += 1

        turn_details.append(
            {
                "turn_index": turn_index,
                "user_message": user_content,
                "expected_answer": exp_content,
                "final_response": final_response,
                "answer_correct": ans_ok,
                "latency_sec": latency,
                "used_tools": used_tools_unique,
                "tool_call_counts": dict(tool_call_counts),  # Count per tool type
                "total_tool_calls": total_tool_calls_this_turn,  # Total calls in this turn
            }
        )

        if verbose:
            print(f"\n[{label} | turn {turn_index}]")
            print(f"  user: {user_content}")
            if exp_content:
                print(f"  expected: {exp_content}")
                print(f"  answer_correct: {ans_ok}")
            print(f"  response: {final_response}")
            if tools_this_turn:
                print(f"  used_tools: {sorted(set(tools_this_turn))}")
            print(f"  latency: {latency:.2f}s")

    return turn_details, tool_counter, answers_expected, answers_correct


def call_agent(
    messages: List[Dict[str, str]],
    server_url: str,
    userid: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    thread_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Call the /invoke endpoint of the agent server.

    Args:
        messages: List of message dicts with 'role' and 'content'
        server_url: Base URL of the agent server
        userid: User ID for memory tools (consistent per user)
        system_prompt: Optional system prompt
        model: Optional model name
        max_search_results: Optional max search results
        timeout: HTTP timeout in seconds
        thread_id: Optional separate thread ID for message history (defaults to userid)
                   Use a unique thread_id per batch to avoid message history pollution.

    Returns (final_response, messages_from_agent).
    """
    payload: Dict[str, Any] = {
        "messages": messages,
        "userid": userid,
    }
    if thread_id:
        payload["thread_id"] = thread_id
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if model:
        payload["model"] = model
    if max_search_results is not None:
        payload["max_search_results"] = max_search_results

    url = server_url.rstrip("/") + "/invoke"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    final_response = data.get("final_response", "")
    messages_out = data.get("messages", [])
    return final_response, messages_out


def reset_short_term_session(
    server_url: str,
    userid: str,
    timeout: float,
    preserve_memory: bool = True,
) -> Dict[str, Any]:
    """
    Reset the short-term session for a user via the /reset endpoint.
    
    Args:
        server_url: Base URL of the agent server
        userid: User identifier
        timeout: HTTP timeout in seconds
        preserve_memory: If True, keep long-term memories intact
        
    Returns:
        Response from the server
    """
    url = f"{server_url.rstrip('/')}/reset/{userid}"
    params = {"preserve_memory": preserve_memory}
    resp = requests.post(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def store_memory(
    server_url: str,
    userid: str,
    key: str,
    content: str,
    timeout: float,
) -> bool:
    """Store a memory via the /memory API."""
    url = f"{server_url.rstrip('/')}/memory/{userid}/store"
    try:
        resp = requests.post(
            url,
            json={"key": key, "content": content},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[WARN] Failed to store memory {key}: {e}")
        return False


def process_session_setup(
    session_setup: Dict[str, Any],
    userid: str,
    *,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Process session_setup by:
    1. Resetting the session (preserving memory) to ensure clean message history
    2. Storing preferences via /memory API
    3. Sending conversation_log messages to the agent to establish context
    
    Returns a dict with setup results.
    """
    setup_result = {
        "preferences_stored": [],
        "preferences_failed": [],
        "conversation_log_processed": False,
    }
    
    # Step 0: Reset session to ensure clean message history (but preserve memory)
    # This prevents issues with incomplete tool calls from previous interactions
    try:
        reset_short_term_session(server_url, userid, timeout, preserve_memory=True)
        if verbose:
            print(f"  [session_setup] Reset session for clean message history")
    except Exception as e:
        if verbose:
            print(f"  [session_setup] Warning: Failed to reset session: {e}")
    
    # Step 1: Store preferences and inform the agent
    stored_preferences = session_setup.get("stored_preferences", [])
    preference_list = []
    
    for pref in stored_preferences:
        key = pref.get("key", "")
        value = pref.get("value", "")
        
        # Convert value to string if it's a list or other type
        if isinstance(value, list):
            content = ", ".join(str(v) for v in value)
        else:
            content = str(value)
        
        # Store as "key: value" format for better searchability
        memory_content = f"{key}: {content}"
        
        success = store_memory(server_url, userid, key, memory_content, timeout)
        if success:
            setup_result["preferences_stored"].append(key)
            preference_list.append(f"{key}: {content}")
            if verbose:
                print(f"  [session_setup] Stored preference: {key} = {content}")
        else:
            setup_result["preferences_failed"].append(key)
            if verbose:
                print(f"  [session_setup] Failed to store preference: {key}")
    
    # Step 1.5: Inform the agent about stored preferences
    # This ensures the agent knows about these preferences and can use them
    # We tell the agent these preferences are already stored, so it doesn't need to store them again
    if preference_list:
        preferences_text = "\n".join(f"- {pref}" for pref in preference_list)
        preferences_message = (
            "I want to share some of my preferences with you. These preferences have already been stored in your memory system, so you don't need to store them again. Just remember them for our conversation:\n\n"
            + preferences_text +
            "\n\nThese preferences are already saved, so please just acknowledge that you understand them."
        )
        try:
            final_response, messages_out = call_agent(
                [{"role": "user", "content": preferences_message}],
                server_url=server_url,
                userid=userid,
                system_prompt=system_prompt,
                model=model,
                max_search_results=max_search_results,
                timeout=timeout,
            )
            if verbose:
                print(f"  [session_setup] Informed agent about {len(preference_list)} preferences")
        except Exception as e:
            if verbose:
                print(f"  [session_setup] Failed to inform agent about preferences: {e}")
            # If informing the agent fails, we still continue with conversation_log
    
    # Step 1.6: Reset session before processing conversation_log to ensure clean message history
    # This prevents issues with incomplete tool calls from preference setup
    # Memory is preserved, so preferences remain accessible via search_memory
    if preference_list or session_setup.get("conversation_log"):
        try:
            reset_short_term_session(server_url, userid, timeout, preserve_memory=True)
            if verbose:
                print(f"  [session_setup] Reset session before conversation_log (memory preserved)")
        except Exception as e:
            if verbose:
                print(f"  [session_setup] Warning: Failed to reset session before conversation_log: {e}")
    
    # Step 2: Process conversation_log to establish context
    # LangGraph maintains conversation history per thread_id (which defaults to userid)
    # So we send each user message sequentially, and LangGraph will maintain the history
    conversation_log = session_setup.get("conversation_log", [])
    if conversation_log:
        for turn in conversation_log:
            role = (turn.get("role") or "").lower()
            content = turn.get("content", "")
            
            if not content:
                continue
            
            if role == "user":
                # Send user message to agent
                # LangGraph will maintain conversation history using the same userid/thread_id
                try:
                    final_response, messages_out = call_agent(
                        [{"role": "user", "content": content}],
                        server_url=server_url,
                        userid=userid,
                        system_prompt=system_prompt,
                        model=model,
                        max_search_results=max_search_results,
                        timeout=timeout,
                    )
                    if verbose:
                        print(f"  [session_setup] Processed conversation log: {content[:50]}...")
                except Exception as e:
                    if verbose:
                        print(f"  [session_setup] Failed to process conversation log: {e}")
            # Note: We skip "agent_expected" messages as they're just for reference
        
        setup_result["conversation_log_processed"] = True
    
    return setup_result


def run_tool_benchmark(
    dataset_name: str,
    path: str,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    limit: Optional[int],
    userid_prefix: str,
    timeout: float,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run the short / medium / long tool benchmarks.

    The JSON file should have the same schema as short.json:

    {
      "test_cases": [
        {
          "id": "test_001_calculator",
          "type": "single_tool",
          "expected_tools": ["calculator"],
          "conversation": [
            {"role": "user", "content": "..."},
            {"role": "agent_expected", "content": "..."}
          ]
        },
        ...
      ]
    }
    """
    data = load_json(path)
    test_cases = data.get("test_cases") or []
    if limit is not None:
        test_cases = test_cases[:limit]

    if not test_cases:
        print(f"[WARN] No test_cases found in {path}")
        return {}

    total = len(test_cases)
    print(f"Running {dataset_name} tool benchmark: {total} cases from {path}")

    subset_tool_correct = 0  # expected_tools ⊆ used_tools
    exact_tool_correct = 0   # expected_tools == used_tools (when non-empty)
    answer_correct = 0
    errors = 0

    tool_expected_counter = Counter()
    tool_used_counter = Counter()

    per_case_results: List[Dict[str, Any]] = []

    for idx, tc in enumerate(test_cases, start=1):
        case_id = tc.get("id", f"{dataset_name}_{idx}")
        expected_tools = set(tc.get("expected_tools") or [])
        conversation = tc.get("conversation") or []
        session_setup = tc.get("session_setup")
        userid = f"{userid_prefix}_{dataset_name}_{idx}"

        if not conversation:
            print(f"[WARN] Skipping {case_id}: empty conversation.")
            continue

        # Process session_setup if present (e.g., for inter_session_context_with_preference tests)
        setup_result = None
        if session_setup:
            if verbose:
                print(f"\n[{case_id}] Processing session_setup...")
            try:
                setup_result = process_session_setup(
                    session_setup,
                    userid,
                    server_url=server_url,
                    system_prompt=system_prompt,
                    model=model,
                    max_search_results=max_search_results,
                    timeout=timeout,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"[WARN] {case_id} session_setup failed: {e}")
                setup_result = {"error": str(e)}

        try:
            turn_details, tool_counter_case, answers_expected, answers_correct_case = (
                process_conversation(
                    conversation,
                    userid=userid,
                    server_url=server_url,
                    system_prompt=system_prompt,
                    model=model,
                    max_search_results=max_search_results,
                    timeout=timeout,
                    verbose=verbose,
                    label=f"{case_id} (conversation)",
                    evaluate_answers=True,
                )
            )
        except Exception as e:
            errors += 1
            print(f"[ERROR] {dataset_name} case {case_id} failed: {e}")
            per_case_results.append(
                {
                    "id": case_id,
                    "session_setup": setup_result,
                    "error": str(e),
                }
            )
            continue

        used_tools = set(tool_counter_case.keys())
        tool_expected_counter.update(expected_tools)
        tool_used_counter.update(used_tools)
        
        # Calculate total tool calls across all turns for this case
        total_tool_calls_case = sum(tool_counter_case.values())

        tools_subset_ok = expected_tools.issubset(used_tools) if expected_tools else True
        tools_exact_ok = used_tools == expected_tools if expected_tools else True

        if tools_subset_ok:
            subset_tool_correct += 1
        if tools_exact_ok:
            exact_tool_correct += 1

        case_answers_correct = (
            answers_expected > 0 and answers_expected == answers_correct_case
        )
        if case_answers_correct:
            answer_correct += 1

        session_meta = metadata_to_dict(
            create_session_metadata(userid, userid, len(turn_details))
        )

        per_case_results.append(
            {
                "id": case_id,
                "expected_tools": sorted(expected_tools),
                "used_tools": sorted(used_tools),
                "tool_call_counts": dict(tool_counter_case),  # Count per tool type across all turns
                "total_tool_calls": total_tool_calls_case,  # Total tool calls across all turns
                "tools_subset_ok": tools_subset_ok,
                "tools_exact_ok": tools_exact_ok,
                "answers_expected": answers_expected,
                "answers_correct": answers_correct_case,
                "all_answers_correct": case_answers_correct,
                "conversation_turns": turn_details,
                "session_metadata": session_meta,
                "session_setup": setup_result,
            }
        )

    print("\n=== Tool benchmark summary ===")
    print(f"Dataset          : {dataset_name}")
    print(f"File             : {path}")
    print(f"Total cases run  : {total}")
    print(f"HTTP errors      : {errors}")
    print(
        f"Tool subset match: {subset_tool_correct}/{total} "
        f"({subset_tool_correct / total * 100:.1f}%)"
    )
    print(
        f"Tool exact match : {exact_tool_correct}/{total} "
        f"({exact_tool_correct / total * 100:.1f}%)"
    )
    print(
        f"Answer match     : {answer_correct}/{total} "
        f"({answer_correct / total * 100:.1f}%)"
    )

    if tool_expected_counter:
        print("\nExpected tool usage counts:")
        for name, count in tool_expected_counter.most_common():
            print(f"  {name}: {count}")
    if tool_used_counter:
        print("\nObserved tool usage counts:")
        for name, count in tool_used_counter.most_common():
            print(f"  {name}: {count}")

    return {
        "dataset": dataset_name,
        "path": path,
        "total_cases": total,
        "errors": errors,
        "tool_subset_correct": subset_tool_correct,
        "tool_exact_correct": exact_tool_correct,
        "answer_correct": answer_correct,
        "per_case_results": per_case_results,
    }


def run_locomo_benchmark(
    path: str,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    limit: Optional[int],
    userid_prefix: str,
    timeout: float,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run the LoCoMo-style benchmark defined in locomo1_converted.json.

    Schema (from the uploaded file):

        {
          "test_cases": [
            {
              "id": "conv-26_session_1",
              "type": "extra_session_context",
              "expected_tools": [],
              "conversation": [
                {
                  "role": "user" | "agent_expected",
                  "speaker": "...",
                  "content": "...",
                  "dia_id": "D1:1"
                },
                ...
              ]
            },
            ...
          ],
          "qa": [
            {
              "question": "...",
              "answer": "...",                  # optional
              "evidence": ["D1:3", ...],
              "category": 1 | 2 | 3 | 4 | 5,
              "adversarial_answer": "..."      # optional
            },
            ...
          ]
        }

    We build an index from dia_id -> (speaker, content), and for each
    QA item we feed the evidence snippets plus the question to the agent.
    """
    data = load_json(path)
    test_cases = data.get("test_cases") or []
    qa_list = data.get("qa") or []

    if not qa_list:
        print(f"[WARN] No 'qa' entries found in {path}")
        return {}

    dia_index: Dict[str, Dict[str, str]] = {}
    for tc in test_cases:
        for turn in tc.get("conversation") or []:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            dia_index[dia_id] = {
                "speaker": turn.get("speaker") or turn.get("role", ""),
                "content": turn.get("content", ""),
            }

    if limit is not None:
        qa_list = qa_list[:limit]

    total = len(qa_list)
    print(f"Running LoCoMo benchmark: {total} QA items from {path}")
    print(f"Indexed {len(dia_index)} dialogue turns by dia_id.\n")

    # Stats.
    answerable_total = 0
    answerable_correct = 0

    adversarial_total = 0
    adversarial_avoided = 0

    errors = 0
    by_category_totals = Counter()
    by_category_correct = Counter()

    per_item_results: List[Dict[str, Any]] = []

    for idx, qa in enumerate(qa_list, start=1):
        question = qa.get("question", "")
        answer_present = "answer" in qa
        expected_answer = qa.get("answer")
        adversarial_answer = qa.get("adversarial_answer")
        category = qa.get("category")
        evidence_ids = qa.get("evidence") or []

        by_category_totals[category] += 1

        ctx_lines: List[str] = []
        for ev in evidence_ids:
            turn = dia_index.get(ev)
            if turn:
                ctx_lines.append(f"{turn['speaker']}: {turn['content']}")
            else:
                ctx_lines.append(f"[Missing evidence: {ev}]")

        context_block = "\n".join(ctx_lines)
        user_text_parts: List[str] = []

        if context_block:
            user_text_parts.append(
                "Use the following background snippets about Caroline, Melanie, "
                "and their lives to answer the question.\n"
            )
            user_text_parts.append("Context:\n" + context_block + "\n")

        user_text_parts.append("Question: " + question)
        user_message_content = "\n".join(user_text_parts)

        messages_in = [{"role": "user", "content": user_message_content}]
        userid = f"{userid_prefix}_locomo_{idx}"

        start = time.time()
        try:
            final_response, messages_out = call_agent(
                messages_in,
                server_url=server_url,
                userid=userid,
                system_prompt=system_prompt,
                model=model,
                max_search_results=max_search_results,
                timeout=timeout,
            )
            latency = time.time() - start

            resp_norm = normalize_text(final_response)

            ans_ok = None
            adv_ok = None

            if answer_present and expected_answer is not None:
                answerable_total += 1
                exp_norm = normalize_text(expected_answer)
                ans_ok = bool(exp_norm) and exp_norm in resp_norm
                if ans_ok:
                    answerable_correct += 1
                    by_category_correct[category] += 1

            if not answer_present and adversarial_answer:
                adversarial_total += 1
                adv_norm = normalize_text(adversarial_answer)
                # "Avoid adversarial answer" = do NOT reproduce that string.
                adv_ok = adv_norm not in resp_norm
                if adv_ok:
                    adversarial_avoided += 1

            per_item_results.append(
                {
                    "index": idx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "adversarial_answer": adversarial_answer,
                    "category": category,
                    "evidence": evidence_ids,
                    "final_response": final_response,
                    "answer_correct": ans_ok,
                    "adversarial_avoided": adv_ok,
                    "latency_sec": latency,
                }
            )

            if verbose:
                print(f"\n[LoCoMo #{idx}]")
                print(f"  Category : {category}")
                print(f"  Evidence : {', '.join(evidence_ids) if evidence_ids else '(none)'}")
                print(f"  Question : {question}")
                if context_block:
                    print("  Context  :")
                    for line in context_block.splitlines():
                        print("    " + line)
                print(f"  Response : {final_response}")
                if answer_present:
                    print(f"  Expected : {expected_answer}")
                    print(f"  answer_correct     : {ans_ok}")
                if not answer_present and adversarial_answer:
                    print(f"  Adversarial target : {adversarial_answer}")
                    print(f"  adversarial_avoided: {adv_ok}")
                print(f"  latency: {latency:.2f}s")

        except Exception as e:
            errors += 1
            print(f"[ERROR] LoCoMo QA #{idx} failed: {e}")
            if verbose:
                per_item_results.append(
                    {
                        "index": idx,
                        "error": str(e),
                        "latency_sec": 0.0,
                    }
                )

    print("\n=== LoCoMo benchmark summary ===")
    print(f"File             : {path}")
    print(f"Total QA items   : {total}")
    print(f"HTTP errors      : {errors}")
    print(
        f"Answerable Qs    : {answerable_total}, "
        f"correct {answerable_correct} "
        f"({(answerable_correct / answerable_total * 100) if answerable_total else 0:.1f}%)"
    )
    print(
        f"Adversarial-only : {adversarial_total}, "
        f"adversarial avoided {adversarial_avoided} "
        f"({(adversarial_avoided / adversarial_total * 100) if adversarial_total else 0:.1f}%)"
    )

    if by_category_totals:
        print("\nPer-category accuracy (answerable questions only):")
        for cat in sorted(by_category_totals.keys()):
            tot = by_category_totals[cat]
            corr = by_category_correct.get(cat, 0)
            pct = (corr / tot * 100) if tot else 0.0
            print(f"  Category {cat}: {corr}/{tot} ({pct:.1f}%)")

    return {
        "dataset": "locomo",
        "path": path,
        "total_items": total,
        "errors": errors,
        "answerable_total": answerable_total,
        "answerable_correct": answerable_correct,
        "adversarial_total": adversarial_total,
        "adversarial_avoided": adversarial_avoided,
        "by_category_totals": dict(by_category_totals),
        "by_category_correct": dict(by_category_correct),
        "per_item_results": per_item_results,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run JSON-defined benchmarks against the CSE291-A agent server."
    )
    parser.add_argument(
        "--dataset",
        choices=["short", "medium", "long", "locomo"],
        required=True,
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark"),
        help="Directory containing benchmark JSON files (default: ./benchmark).",
    )
    parser.add_argument(
        "--locomo-file",
        default=None,
        help="Path to the LoCoMo JSON file "
             "(default: <benchmark-dir>/locomo1_converted.json).",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("BENCHMARK_SERVER_URL", "http://localhost:8000"),
        help="Base URL of the agent server (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Model name/ID to pass through to the server (default: env MODEL or server default).",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get(
            "SYSTEM_PROMPT", "You are a helpful AI assistant."
        ),
        help="System prompt string (default: env SYSTEM_PROMPT or a simple helper prompt).",
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=int(os.environ.get("MAX_SEARCH_RESULTS", "10")),
        help="max_search_results to send with each request (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only run the first N test cases / QA items.",
    )
    parser.add_argument(
        "--userid-prefix",
        default="benchmark",
        help="Prefix for the userid field used when calling the agent.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per /invoke call in seconds (default: %(default)s). "
             "Increased default for web search operations.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-case details.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results JSON (default: benchmark_results_{dataset}.json).",
    )

    args = parser.parse_args(argv)

    benchmark_dir = args.benchmark_dir

    if args.dataset == "locomo":
        locomo_file = args.locomo_file or os.path.join(
            benchmark_dir, "locomo1_converted.json"
        )
        stats = run_locomo_benchmark(
            path=locomo_file,
            server_url=args.server_url,
            system_prompt=args.system_prompt,
            model=args.model,
            max_search_results=args.max_search_results,
            limit=args.limit,
            userid_prefix=args.userid_prefix,
            timeout=args.timeout,
            verbose=args.verbose,
        )
    else:
        dataset_file = os.path.join(benchmark_dir, f"{args.dataset}.json")
        stats = run_tool_benchmark(
            dataset_name=args.dataset,
            path=dataset_file,
            server_url=args.server_url,
            system_prompt=args.system_prompt,
            model=args.model,
            max_search_results=args.max_search_results,
            limit=args.limit,
            userid_prefix=args.userid_prefix,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    if not stats:
        sys.exit(1)

    # Save results to JSON file
    output_file = args.output
    if output_file is None:
        output_file = f"benchmark_results_{args.dataset}.json"
    
    # Add metadata to results
    results_with_meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": args.dataset,
        "server_url": args.server_url,
        "model": args.model,
        "system_prompt": args.system_prompt,
        "max_search_results": args.max_search_results,
        "limit": args.limit,
        **stats,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
