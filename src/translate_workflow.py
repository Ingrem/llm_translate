import time


class TranslateWorkflow:
    """
    Workflow for translating Japanese text into Russian using an LLM,
    followed by light editing of the translated text.
    """

    def __init__(self, llm, max_tokens=256, temperature=0.3, show_prompt=False):
        """
        Initialize TranslateWorkflow.

        Args:
            llm: LLM client with method generate_with_retry(prompt, max_tokens, temperature, resp_name).
            max_tokens (int, optional): Maximum number of tokens for generation. Defaults to 256.
            temperature (float, optional): Sampling temperature (0.0 = deterministic). Defaults to 0.3.
            show_prompt (bool, optional): If True, print prompts sent to the LLM for debugging. Defaults to False.
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.show_prompt = show_prompt

    def _print_prompt(self, prompt: str, label: str = "PROMPT") -> None:
        """Optionally print the LLM prompt for debugging."""
        if self.show_prompt:
            sep = "=" * 40
            print(f"{sep}{label}{sep}")
            print(prompt)
            print(f"{sep}{label} END{sep}")

    def run_translate(self, line: str, context: str = "") -> str:
        """
        Main entry point. Takes one line of text and optional context,
        returns translated and edited text.
        """
        start = time.time()
        translated = self._translate_with_llm(line.strip(), context)
        print(f"done in {time.time() - start:.1f}s\n")
        return translated

    def _translate_with_llm(self, line: str, context: str) -> str:
        """Send translation prompt to LLM and pass result through editing step."""
        print(f"original: {line}")

        prompt = (
f"""Ты — литературный переводчик. Переведи на русский язык.
Фразы внутри 「」 — это прямая речь, переводи их как прямую речь.  
Учитывай контекст вокруг текста и описание персонажей для правильной передачи смысла. 
Если в тексте нет японского, тогда **повтори его дословно**.
Вывод строго: одна строка перевода, без комментариев.
{context}
Пример перевода:
Японский: アンナは毎あさ七時に起きます。
Русский: Анна встает в семь утра каждое утро.

Японский: ......
Русский: ......

Переведи:
Японский: {line}
Русский:""")

        self._print_prompt(prompt, label="TRANSLATE PROMPT")

        raw_translation = self.llm.generate_with_retry(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            resp_name="translation"
        )

        return self._edit_with_llm(raw_translation)


    def _edit_with_llm(self, line: str) -> str:
        """Send Russian translation to LLM for light editing."""
        prompt = (
f"""Ты — литературный редактор. 
Твоя задача — сделать исходный русский текст грамматически правильным, легким для чтения и убрать сложные обороты. 
Сохраняй точный смысл оригинала, не добавляй деталей или эмоций, которых там нет.
Если текст не требует изменений, **повтори его дословно** — **включая все знаки препинания, пробелы и оформление**.

Вывод: одна строка, без комментариев.

Примеры:
Текст: Ваа, свет луны отражается в тихой воде пруда, делая ночь казаться ещё более долгой.
Вывод: Вау, лунный свет, отражаясь в тихой воде пруда, делал ночь бесконечно долгой.

Текст: ......
Вывод: ......

Текст: {line}
Вывод:""")

        self._print_prompt(prompt, label="EDIT PROMPT")

        return self.llm.generate_with_retry(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            resp_name="edit"
        )
