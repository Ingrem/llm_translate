import time
from src.kag_workflow import KagWorkflow
from src.translate_workflow import TranslateWorkflow


class FileWorkflow:
    """
    Workflow for translating script files line by line.

    Uses:
      - KagWorkflow for building context,
      - TranslateWorkflow for actual translation with LLM.
    """

    def __init__(self, llm, show_prompts: bool = False, kag_enabled: bool = False):
        """
        Args:
            llm: LLM wrapper instance.
            show_prompts (bool): Whether to print prompts before sending to LLM.
        """
        self.kag = KagWorkflow()
        self.kag_enabled = kag_enabled
        self.translate = TranslateWorkflow(llm, show_prompt=show_prompts)
        self.file_data = {}

    def _print_progress(self, line_num: int) -> None:
        """
        Print translation progress with ETA.

        Args:
            line_num (int): Current line index.
        """
        elapsed = time.time() - self.file_data["start_file_translation"]
        total = self.file_data["all_lines_len"]

        eta = 0
        if line_num > 0:
            eta = elapsed / line_num * total - elapsed

        percent = (line_num * 100) // total
        print(
            f"Working with line {line_num}/{total} ({percent}%) "
            f"total time: {elapsed / 60:.1f} min, ETA: {eta / 60:.1f} min"
        )

    def _translate_line(self, line_num: int, line: str) -> str:
        """
        Translate a single line with context.
        """
        self._print_progress(line_num)

        context = ""
        if self.kag_enabled:
            context = self.kag.build_context(self.file_data["all_lines"], line_num)

        return self.translate.run_translate(line, context)

    def _segregate_and_translate_lines(self, all_lines: list[str]) -> list[str]:
        """
        Translate all lines, skipping empty ones.
        """
        result = []
        for line_num, line in enumerate(all_lines):
            if line.strip() == "":
                result.append(line)
            else:
                result.append(f"{self._translate_line(line_num, line)}\n")
        return result

    def translate_one_file(self, input_path: str, output_path: str, encoding: str = "UTF-8") -> None:
        """
        Translate an entire file line by line.

        Args:
            input_path (str | Path): Path to input file.
            output_path (str | Path): Path to save translated file.
            encoding (str): File encoding. Default is "shift_jis".
        """
        with open(input_path, "r", encoding=encoding, errors="ignore") as f:
            all_lines = f.readlines()

        self.file_data = {
            "start_file_translation": time.time(),
            "all_lines": all_lines,
            "all_lines_len": len(all_lines),
        }

        result = self._segregate_and_translate_lines(all_lines)

        print("Translation done successfully! (100%)")
        elapsed = time.time() - self.file_data["start_file_translation"]
        print(f"done in {elapsed / 60:.1f} min")

        with open(output_path, "w", encoding=encoding) as f:
            f.writelines(result)
