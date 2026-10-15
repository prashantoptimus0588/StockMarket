"""LangGraph nodes, assessment workflow, report persistence, and approval routing."""
import asyncio
import json
import logging
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from database.repository import get_assessment_by_thread_id, record_decision, save_assessment
from models.models import FundamentalsData, PriceData
from report.report import build_multi_report, build_report, multi_report_to_dict, report_to_dict
from chat.assessment_logic import (build_assessments, enforce_stance, fallback_explanation, format_facts, format_verdict, parse_tool_result, render_answer, validate_explanation)
from chat.config import FETCH_RETRY_DELAY, MAX_REPLANS, MAX_STEPS, MAX_TICKERS, TOOL_FUNDAMENTALS, TOOL_NEWS, TOOL_PRICE, TOOL_SEARCH
from chat.llm_utils import ask
from chat.prompts import EXPLAIN_SYSTEM, build_general_prompt, build_plan_prompt, build_plan_user_message, build_resolve_prompt
from chat.report_files import REPORT_JSON_DIR, REPORT_MD_DIR, safe_name, save_report_files
from chat.schemas import Explanation, Target
from chat.state import AssessmentState
from chat.tool_helpers import call_tool, get_field
log = logging.getLogger("agent")

def build_graph(rt) -> StateGraph:

    # ---- plan -------------------------------------------------------------
    async def plan_node(state: AssessmentState) -> dict:
        try:
            plan = await ask(rt.planner, [SystemMessage(content=build_plan_prompt()),
                                          HumanMessage(content=build_plan_user_message(
                                              state.query, state.rejected_targets,
                                              state.approval_comment, state.replan_hint))], "plan")
        except Exception:
            log.warning("planner failed; routing to the general agent")
            return {"route": "general", "replan_pending": False}
        targets, seen = [], set()
        for t in plan.targets:
            key = (t.ticker or t.company).strip().upper()
            if key and key not in seen:
                seen.add(key)
                targets.append(Target(company=t.company.strip(), ticker=(t.ticker or "").strip() or None))
        targets = targets[:MAX_TICKERS]
        route = "stock_analysis" if plan.route == "stock_analysis" and targets and state.prefs else "general"
        log.info("plan: route=%s targets=%s news=%s", route, [(t.company, t.ticker) for t in targets], plan.wants_news)
        return {"route": route, "targets": targets, "wants_news": plan.wants_news, "replan_pending": False}

    def route_after_plan(state: AssessmentState) -> str:
        return "resolve" if state.route == "stock_analysis" else "agent"

    # ---- resolve tickers (search tool in code, LLM only picks the listing) --
    async def resolve_node(state: AssessmentState) -> dict:
        need = [t for t in state.targets if not t.ticker]
        if not need:
            return {}
        search = rt.tools_by_name[TOOL_SEARCH]
        results = await asyncio.gather(*(call_tool(search, t.company) for t in need), return_exceptions=True)
        blocks = []
        for t, r in zip(need, results):
            text = f"ERROR: {type(r).__name__}: {r}" if isinstance(r, Exception) else r
            blocks.append(f"### {t.company}\n{text[:3000]}")
        res = await ask(rt.resolver, [SystemMessage(content=build_resolve_prompt()),
                                      HumanMessage(content="\n\n".join(blocks))], "resolve_ticker")
        by_name = {c.company.strip().lower(): c.ticker for c in res.choices}
        by_index = [c.ticker for c in res.choices] if len(res.choices) == len(need) else []
        warnings = list(state.warnings)
        resolved: dict = {}
        for i, t in enumerate(need):
            tk = by_name.get(t.company.strip().lower()) or (by_index[i] if by_index else None)
            if tk and tk.strip():
                resolved[t.company] = tk.strip()
            else:
                warnings.append(f"Could not resolve a ticker for '{t.company}'.")
        targets = [Target(company=t.company, ticker=t.ticker or resolved.get(t.company))
                   for t in state.targets if t.ticker or t.company in resolved]
        return {"targets": targets, "warnings": warnings}

    # ---- fetch data in parallel (no LLM involved) -----------------------------
    async def fetch_node(state: AssessmentState) -> dict:
        price_tool = rt.tools_by_name[TOOL_PRICE]
        fund_tool = rt.tools_by_name[TOOL_FUNDAMENTALS]
        news_tool = rt.tools_by_name.get(TOOL_NEWS)
        warnings = list(state.warnings)
        if state.wants_news and news_tool is None:
            warnings.append("The news tool is not available on the MCP server.")

        async def fetch_one(t: Target):
            coros = [call_tool(price_tool, t.ticker), call_tool(fund_tool, t.ticker)]
            if state.wants_news and news_tool is not None:
                coros.append(call_tool(news_tool, f"{t.company} stock"))
            return t, await asyncio.gather(*coros, return_exceptions=True)

        results = await asyncio.gather(*(fetch_one(t) for t in state.targets))
        specs = {"price": (price_tool, PriceData), "fundamentals": (fund_tool, FundamentalsData)}
        prices, funds, news, flagged = {}, {}, {}, []
        buckets = {"price": prices, "fundamentals": funds}
        symbols: dict = {}
        failed: dict = {}                      # (ticker key, label) -> reason the FETCH failed
        for t, res in results:
            key = t.ticker.strip().upper()
            symbols[key] = t.ticker
            for label, raw in zip(("price", "fundamentals"), res[:2]):
                if isinstance(raw, str) and "DATA QUALITY WARNING" in raw and key not in flagged:
                    flagged.append(key)
                    warnings.append(f"{key}: the data source flagged its data as unreliable.")
                obj, reason = parse_tool_result(raw, specs[label][1], key, label)
                if reason:
                    failed[(key, label)] = reason
                else:
                    buckets[label][key] = obj
            if len(res) > 2:
                if isinstance(res[2], Exception):
                    warnings.append(f"{key}: news fetch failed ({type(res[2]).__name__}).")
                else:
                    news[key] = res[2][:4000]

        # one retry for anything whose FETCH failed (Yahoo throttling is usually short-lived)
        if failed:
            log.warning("fetch: %s failed, retrying once in %ss",
                        [f"{k}/{l}" for k, l in failed], FETCH_RETRY_DELAY)
            await asyncio.sleep(FETCH_RETRY_DELAY)
            pairs = list(failed)
            redo = await asyncio.gather(*(call_tool(specs[l][0], symbols[k]) for k, l in pairs),
                                        return_exceptions=True)
            for (k, l), raw in zip(pairs, redo):
                obj, reason = parse_tool_result(raw, specs[l][1], k, l)
                if reason:
                    failed[(k, l)] = reason
                else:
                    buckets[l][k] = obj
                    del failed[(k, l)]
                    log.info("fetch: retry fixed %s %s", k, l)

        fetch_failures: dict = {}
        for (k, l), reason in failed.items():
            note = f"{'price data' if l == 'price' else 'fundamentals'} could not be fetched ({reason})"
            fetch_failures[k] = f"{fetch_failures[k]}; {note}" if k in fetch_failures else note
            log.error("fetch FAILED for %s: %s", k, note)
        return {"prices": prices, "fundamentals": funds, "news": news, "flagged_tickers": flagged,
                "warnings": warnings, "fetch_failures": fetch_failures}

    # ---- deterministic engines ------------------------------------------------
    def risk_engine_node(state: AssessmentState) -> dict:
        warnings = list(state.warnings)
        assessments = build_assessments(state.prices, state.fundamentals, state.prefs, warnings, state.fetch_failures)
        return {"assessments": assessments, "warnings": warnings}

    # ---- explain (LLM, structured, no tools) + validation ----------------------
    async def explain_node(state: AssessmentState) -> dict:
        if not state.assessments:
            if state.fetch_failures:
                lines = ["I could not produce an assessment because the market data could not be fetched. "
                         "This is a data-fetch failure (often Yahoo rate limiting), not a verdict on the stock. "
                         "Try again in a minute."]
                lines += [f"- {tk}: {why[:1].upper() + why[1:]} (retried once)."
                          for tk, why in state.fetch_failures.items()]
            else:
                lines = ["I could not produce an assessment because no ticker had complete, readable data."]
            lines += [f"- {w}" for w in state.warnings]
            msg = "\n".join(lines)
            out = {"messages": [AIMessage(content=msg)], "answer": msg, "replan_pending": False, "replan_hint": msg}
            if state.rejected_targets:
                seen = list(state.rejected_targets)
                for t in state.targets:
                    label = f"{t.company} ({t.ticker})" if t.ticker else t.company
                    if label and label not in seen:
                        seen.append(label)
                out["rejected_targets"] = seen
                if state.replan_count < MAX_REPLANS:
                    log.info("replan produced no assessment; trying another plan (%s/%s)",
                             state.replan_count + 1, MAX_REPLANS)
                    out["replan_pending"] = True
                    out["replan_count"] = state.replan_count + 1
            return out

        flagged = {t.upper() for t in state.flagged_tickers}
        blocks = [format_verdict(tk, state.prefs, it["verdict"], it["returns"], it["combined"])
                  + "\n\n" + format_facts(it["price"], it["fundamentals"])
                  for tk, it in state.assessments.items()]
        context = "\n\n".join(blocks)
        if state.warnings:
            context += "\n\nDATA WARNINGS:\n" + "\n".join(f"- {w}" for w in state.warnings)
        if state.fetch_failures:
            context += "\n\nNOT ASSESSED (data fetch failed; do not rate or compare these):\n" + "\n".join(
                f"- {tk}: {why}" for tk, why in state.fetch_failures.items())
        if state.news:
            context += "\n\n" + "\n\n".join(f"NEWS for {tk}:\n{txt}" for tk, txt in state.news.items())
        user_q = state.query
        if state.approval_comment:
            user_q += f"\n\nThe user rejected the previous report and asked instead: {state.approval_comment}"
        messages = [SystemMessage(content=EXPLAIN_SYSTEM),
                    HumanMessage(content=f"User question: {user_q}\n\n{context}")]

        try:
            exp: Explanation = await ask(rt.explainer, messages, "explain")
            problems = validate_explanation(exp, state.assessments, flagged)
            if problems:
                log.warning("explain: validation problems, retrying once: %s", problems)
                messages.append(HumanMessage(content="Fix these problems and answer again:\n- " + "\n- ".join(problems)))
                exp = await ask(rt.explainer, messages, "explain_retry")
        except Exception:
            log.exception("explain LLM failed; using engine-only narrative")
            exp = fallback_explanation(state.assessments, flagged)
        exp = enforce_stance(exp, state.assessments, flagged)

        news_missing = state.wants_news and not state.news
        answer = render_answer(exp, state.warnings, state.wants_news, news_missing, state.fetch_failures)
        return {"messages": [AIMessage(content=answer)], "answer": answer}

    def route_after_explain(state: AssessmentState) -> str:
        if state.assessments:
            return "report"
        if state.replan_pending:
            return "plan"
        if state.previous_report_markdown:
            return "restore"
        return "end"

    # ---- report + approval ------------------------------------------------------
    def report_node(state: AssessmentState, config: RunnableConfig) -> dict:
        """Build report artifacts and persist one assessment per graph thread (idempotent)."""
        thread_id = config["configurable"]["thread_id"]

        if state.report_saved and state.assessment_id is not None:
            log.info("report already saved as assessment #%s; skipping duplicate save", state.assessment_id)
            return {k: getattr(state, k) for k in (
                "report_markdown", "report_json", "assessment_id", "approval_status",
                "approval_comment", "report_saved", "report_md_path", "report_json_path")}

        # Recovery: a prior run may have committed the DB row but not checkpointed the node output.
        # After a reject we insert a new row; do not reuse a reviewed assessment for this thread.
        existing = get_assessment_by_thread_id(thread_id)
        if existing is not None and existing.get("approval_status") == "pending":
            existing_md = existing.get("report_markdown") or ""
            existing_json = existing.get("report_json")
            md_path = json_path = None
            if existing_md and REPORT_MD_DIR.exists():
                for cand in REPORT_MD_DIR.glob("*.md"):
                    try:
                        if cand.read_text(encoding="utf-8") == existing_md:
                            md_path = cand
                            break
                    except OSError:
                        continue
            if isinstance(existing_json, dict) and REPORT_JSON_DIR.exists():
                for cand in REPORT_JSON_DIR.glob("*.json"):
                    try:
                        if json.loads(cand.read_text(encoding="utf-8")) == existing_json:
                            json_path = cand
                            break
                    except (OSError, json.JSONDecodeError):
                        continue
            log.info("reusing assessment #%s for this thread", existing["id"])
            return {
                "report_markdown": existing_md, "report_json": existing_json,
                "assessment_id": existing["id"],
                "approval_status": existing.get("approval_status", "pending"),
                "approval_comment": existing.get("approval_comment"),
                "report_saved": True,
                "report_md_path": str(md_path) if md_path else None,
                "report_json_path": str(json_path) if json_path else None,
            }

        assessments = list(state.assessments.values())
        if not assessments:
            raise ValueError("Cannot generate a report: no ticker had both valid price and fundamentals data.")

        answer = state.answer
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tickers = [it["price"].ticker for it in assessments]

        if len(assessments) > 1:
            report_markdown = build_multi_report(query=state.query, prefs=state.prefs,
                                                 assessments=assessments, llm_answer=answer,
                                                 fetch_failures=state.fetch_failures)
            report_json = multi_report_to_dict(query=state.query, prefs=state.prefs,
                                               assessments=assessments, llm_answer=answer,
                                                 fetch_failures=state.fetch_failures)
            ticker_label = "_vs_".join(safe_name(t) for t in tickers)
            risk_label = return_label = "multiple"
            combined_label = "Multi-stock comparison: " + "; ".join(
                f"{it['price'].ticker}: {it['combined'].summary}" for it in assessments)
        else:
            item = assessments[0]
            kwargs = {"query": state.query, "prefs": state.prefs, "price": item["price"],
                      "fundamentals": item["fundamentals"], "verdict": item["verdict"],
                      "returns": item["returns"], "combined": item["combined"], "llm_answer": answer,
                      "fetch_failures": state.fetch_failures}
            report_markdown = build_report(**kwargs)
            report_json = report_to_dict(**kwargs)
            ticker_label = item["price"].ticker
            risk_label = str(item["verdict"].result)
            return_label = str(item["returns"].level)
            combined_label = str(item["combined"].summary)

            payload = item["price"].model_dump() if hasattr(item["price"], "model_dump") else {}
            currency = payload.get("currency") or payload.get("currency_code") or payload.get("currency_symbol")
            if currency:
                report_markdown = report_markdown.replace(
                    "_Prices in listing currency (not reported)._",
                    f"_Prices in {currency} (listing currency)._")
                if isinstance(report_json, dict):
                    report_json.setdefault("market_data", {})
                    if isinstance(report_json["market_data"], dict):
                        report_json["market_data"]["currency"] = currency

        md_path, json_path = save_report_files(report_markdown, report_json, ticker_label, stamp)
        log.info("saved report %s and %s", md_path, json_path)
        saved = save_assessment(
            thread_id=thread_id, ticker=ticker_label,
            profile=str(state.prefs.profile), horizon=str(state.prefs.horizon),
            risk_verdict=risk_label, return_potential=return_label, combined_verdict=combined_label,
            report_markdown=report_markdown, report_json=report_json)
        assessment_id = getattr(saved, "id", saved)
        log.info("report saved as assessment #%s; approval pending", assessment_id)
        return {
            "report_markdown": report_markdown, "report_json": report_json,
            "assessment_id": assessment_id, "approval_status": "pending", "approval_comment": None,
            "report_saved": True, "report_md_path": str(md_path), "report_json_path": str(json_path),
        }

    def approval_node(state: AssessmentState) -> dict:
        """Pause until the user approves or rejects. Nothing with side effects runs before interrupt()."""
        decision = interrupt({
            "type": "report_approval",
            "assessment_id": state.assessment_id,
            "ticker": ", ".join(state.assessments.keys()),
            "report_markdown": state.report_markdown,
            "message": "Review this report and approve or reject it.",
        })
        if not isinstance(decision, dict):
            raise ValueError("Expected an approval decision payload.")
        status = str(decision.get("decision", "")).strip().lower()
        comment = decision.get("comment")
        if status not in {"approved", "rejected"}:
            raise ValueError("Decision must be 'approved' or 'rejected'.")
        comment_text = comment.strip() if isinstance(comment, str) and comment.strip() else None
        record_decision(assessment_id=state.assessment_id, decision=status, comment=comment_text)
        log.info("assessment #%s: %s", state.assessment_id, status)
        out = {
            "approval_status": status,
            "approval_comment": comment_text,
            "replan_pending": False,
        }
        # Reject: send the planner a new set of companies (capped). Approve: stop for QnA.
        if status == "rejected" and state.replan_count < MAX_REPLANS:
            seen = list(state.rejected_targets)
            for t in state.targets:
                label = f"{t.company} ({t.ticker})" if t.ticker else t.company
                if label and label not in seen:
                    seen.append(label)
            log.info("reject: replanning alternatives (%s/%s); skipped=%s",
                     state.replan_count + 1, MAX_REPLANS, seen)
            out.update({
                "replan_pending": True,
                "replan_count": state.replan_count + 1,
                "rejected_targets": seen,
                "replan_hint": "",
                "previous_report_markdown": state.report_markdown or state.previous_report_markdown,
                "previous_report_json": state.report_json if state.report_json is not None else state.previous_report_json,
                "previous_assessment_id": state.assessment_id or state.previous_assessment_id,
                "previous_report_md_path": state.report_md_path or state.previous_report_md_path,
                "previous_report_json_path": state.report_json_path or state.previous_report_json_path,
                "route": "",
                "targets": [],
                "wants_news": False,
                "prices": {},
                "fundamentals": {},
                "news": {},
                "flagged_tickers": [],
                "warnings": [],
                "fetch_failures": {},
                "assessments": {},
                "answer": "",
                "report_markdown": "",
                "report_json": None,
                "assessment_id": None,
                "report_saved": False,
                "report_md_path": None,
                "report_json_path": None,
            })
        elif status == "rejected":
            log.info("reject: replan limit reached (%s); ending for QnA on the last report", MAX_REPLANS)
        return out

    def route_after_approval(state: AssessmentState) -> str:
        return "plan" if state.replan_pending else "end"

    def restore_report_node(state: AssessmentState) -> dict:
        """Replan could not produce a new report; put the last good one back for the API/QnA."""
        log.warning("replan produced no assessment; restoring previous report #%s",
                    state.previous_assessment_id)
        note = state.answer or "Could not assess an alternative. Keeping the previous report."
        return {
            "report_markdown": state.previous_report_markdown,
            "report_json": state.previous_report_json,
            "assessment_id": state.previous_assessment_id,
            "report_saved": True,
            "report_md_path": state.previous_report_md_path,
            "report_json_path": state.previous_report_json_path,
            "approval_status": "rejected",
            "replan_pending": False,
            "answer": note,
            "messages": [AIMessage(content=note)],
        }

    # ---- general agent loop (weather, general news, ...) -------------------------
    async def agent_node(state: AssessmentState) -> dict:
        msgs = [SystemMessage(content=build_general_prompt())] + list(state.messages)
        resp = await ask(rt.agent_llm, msgs, f"agent step {state.current_step + 1}")
        return {"messages": [resp], "current_step": state.current_step + 1}

    def route_after_agent(state: AssessmentState) -> str:
        last = state.messages[-1] if state.messages else None
        if getattr(last, "tool_calls", None):
            if state.current_step >= MAX_STEPS:
                log.warning("step limit reached while the model still wanted tools")
                return "end"
            return "tools"
        return "end"

    g = StateGraph(AssessmentState)
    g.add_node("plan_node", plan_node)
    g.add_node("resolve_node", resolve_node)
    g.add_node("fetch_node", fetch_node)
    g.add_node("risk_engine_node", risk_engine_node)
    g.add_node("explain_node", explain_node)
    g.add_node("report_node", report_node)
    g.add_node("approval_node", approval_node)
    g.add_node("restore_report_node", restore_report_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(rt.tools))

    g.add_edge(START, "plan_node")
    g.add_conditional_edges("plan_node", route_after_plan, {"resolve": "resolve_node", "agent": "agent_node"})
    g.add_edge("resolve_node", "fetch_node")
    g.add_edge("fetch_node", "risk_engine_node")
    g.add_edge("risk_engine_node", "explain_node")
    g.add_conditional_edges("explain_node", route_after_explain, {
        "report": "report_node", "plan": "plan_node", "restore": "restore_report_node", "end": END,
    })
    g.add_edge("restore_report_node", END)
    g.add_edge("report_node", "approval_node")
    g.add_conditional_edges("approval_node", route_after_approval, {"plan": "plan_node", "end": END})
    g.add_conditional_edges("agent_node", route_after_agent, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent_node")
    return g

