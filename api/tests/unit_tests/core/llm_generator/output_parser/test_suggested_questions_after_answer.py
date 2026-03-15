from unittest.mock import patch

import pytest

from core.llm_generator.output_parser.suggested_questions_after_answer import SuggestedQuestionsAfterAnswerOutputParser
from core.llm_generator.prompts import _DEFAULT_SUGGESTED_QUESTIONS_AFTER_ANSWER_PROMPT


class TestSuggestedQuestionsAfterAnswerOutputParser:
    @pytest.fixture
    def parser(self):
        return SuggestedQuestionsAfterAnswerOutputParser()

    def test_get_format_instructions_returns_default_prompt(self, parser):
        with patch(
            "core.llm_generator.output_parser.suggested_questions_after_answer.SUGGESTED_QUESTIONS_AFTER_ANSWER_INSTRUCTION_PROMPT",
            _DEFAULT_SUGGESTED_QUESTIONS_AFTER_ANSWER_PROMPT,
        ):
            instructions = parser.get_format_instructions()
            assert instructions == _DEFAULT_SUGGESTED_QUESTIONS_AFTER_ANSWER_PROMPT

    def test_get_format_instructions_uses_custom_env_prompt(self, parser):
        custom_prompt = "Generate 5 questions in JSON format: [q1, q2, q3, q4, q5]"
        with patch(
            "core.llm_generator.output_parser.suggested_questions_after_answer.SUGGESTED_QUESTIONS_AFTER_ANSWER_INSTRUCTION_PROMPT",
            custom_prompt,
        ):
            instructions = parser.get_format_instructions()
            assert instructions == custom_prompt

    def test_parse_valid_json_array(self, parser):
        text = '["What is Dify?", "How to use it?", "What features exist?"]'
        questions = parser.parse(text)
        assert len(questions) == 3
        assert questions[0] == "What is Dify?"
        assert questions[1] == "How to use it?"
        assert questions[2] == "What features exist?"

    def test_parse_json_embedded_in_text(self, parser):
        text = 'Here are the questions:\n["What is Dify?", "How to use it?"]\nEnd.'
        questions = parser.parse(text)
        assert len(questions) == 2
        assert questions[0] == "What is Dify?"

    def test_parse_empty_array(self, parser):
        questions = parser.parse("[]")
        assert questions == []

    def test_parse_invalid_json(self, parser):
        questions = parser.parse("[invalid json")
        assert questions == []

    def test_parse_non_string_elements_filtered(self, parser):
        text = '["Valid question?", 42, null, "Another question?"]'
        questions = parser.parse(text)
        assert len(questions) == 2
        assert questions[0] == "Valid question?"
        assert questions[1] == "Another question?"

    def test_parse_no_array_in_text(self, parser):
        questions = parser.parse("There are no questions here.")
        assert questions == []

    def test_parse_empty_string(self, parser):
        questions = parser.parse("")
        assert questions == []

    def test_parse_whitespace_only(self, parser):
        questions = parser.parse("   ")
        assert questions == []

    def test_parse_multiline_array(self, parser):
        text = '[\n  "What can you do?",\n  "How do I start?"\n]'
        questions = parser.parse(text)
        assert len(questions) == 2
        assert questions[0] == "What can you do?"
        assert questions[1] == "How do I start?"

    def test_parse_chinese_questions(self, parser):
        text = '["你可以做什么?", "如何使用?", "有什么功能?"]'
        questions = parser.parse(text)
        assert len(questions) == 3
        assert questions[0] == "你可以做什么?"
