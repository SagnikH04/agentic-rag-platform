import re
from time import perf_counter

from app.agents.memory import ConversationMemory
from app.agents.tools import detect_prompt_injection, retrieve_documents, sentence_support, verify_citations
from app.models.schemas import QueryResponse


class AgentOrchestrator:
    def __init__(self) -> None:
        self.memory = ConversationMemory()

    async def answer(self, question: str, session_id: str, filters: dict, top_k: int) -> QueryResponse:
        history = await self.memory.recent(session_id)
        rewritten_question = self._rewrite_query(question, history)
        started = perf_counter()
        citations = await retrieve_documents(rewritten_question, top_k, filters)
        retrieval_ms = round((perf_counter() - started) * 1000, 2)
        safe_citations = [c for c in citations if not detect_prompt_injection(c.text)]
        safety_flags = []
        if detect_prompt_injection(question):
            safety_flags.append("query_contains_instruction_like_text")
        if len(safe_citations) != len(citations):
            safety_flags.append("retrieved_instruction_like_content_removed")

        if not safe_citations:
            answer = "I could not find enough grounded context to answer confidently."
            await self.memory.append(session_id, "user", question)
            await self.memory.append(session_id, "assistant", answer)
            return QueryResponse(
                answer=answer,
                citations=[],
                confidence=0.0,
                rewritten_query=rewritten_question,
                retrieval_ms=retrieval_ms,
                session_id=session_id,
                safety_flags=safety_flags,
            )

        answer = self._synthesize(question, safe_citations)
        citation_confidence = verify_citations(answer, safe_citations)
        support_confidence = sentence_support(answer, safe_citations)
        retrieval_confidence = sum(max(0.0, min(1.0, item.score)) for item in safe_citations[:3]) / min(3, len(safe_citations))
        confidence = round(0.45 * citation_confidence + 0.35 * support_confidence + 0.20 * retrieval_confidence, 3)
        if support_confidence < 0.5:
            safety_flags.append("answer_has_limited_evidence_support")

        await self.memory.append(session_id, "user", question)
        await self.memory.append(session_id, "assistant", answer)
        return QueryResponse(
            answer=answer,
            citations=safe_citations,
            confidence=confidence,
            rewritten_query=rewritten_question,
            retrieval_ms=retrieval_ms,
            session_id=session_id,
            safety_flags=safety_flags,
        )

    def _rewrite_query(self, question: str, history: list[dict[str, str]]) -> str:
        words = question.split()
        previous_questions = [turn["content"] for turn in history if turn["role"] == "user"]
        if len(words) > 8 or not previous_questions:
            return question
        return f"{question} Context from the preceding user question: {previous_questions[-1][:600]}"

    def _synthesize(self, question: str, citations) -> str:
        question_terms = {
            token
            for token in re.findall(r"\b[\w-]+\b", question.lower())
            if len(token) > 2
        }
        candidates: list[tuple[float, int, str]] = []
        for citation_index, citation in enumerate(citations, 1):
            for sentence in re.split(r"(?<=[.!?])\s+", citation.text):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                terms = set(re.findall(r"\b[\w-]+\b", sentence.lower()))
                overlap = len(question_terms & terms) / max(1, len(question_terms))
                score = overlap + min(0.4, citation.score * 0.4)
                candidates.append((score, citation_index, sentence))

        selected: list[tuple[int, str]] = []
        seen = set()
        for _, citation_index, sentence in sorted(candidates, reverse=True):
            fingerprint = re.sub(r"\W+", "", sentence.lower())[:180]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append((citation_index, sentence))
            if len(selected) == 3:
                break
        if not selected:
            return "I found related sources, but not enough extractable evidence to give a grounded answer."
        return "\n".join(f"[{index}] {sentence}" for index, sentence in selected)
