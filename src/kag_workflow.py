import json
from config import ROOT_DIR


class KagWorkflow:
    """
    Workflow for building translation context from KAG script lines.

    Provides utilities:
      - surrounding scene context (window of lines),
      - dictionary of relevant character name translations,
      - description of the speaking character or narration type.
    """

    def __init__(self, window_size=4):
        """
        Initialize KagWorkflow.

        Args:
            window_size (int, optional): Number of lines before and after the target line
                                         included in the scene context. Defaults to 4.
        """
        self.window_size = window_size
        kag_data_path = f"{ROOT_DIR}/src/kag_data"

        with open(f"{kag_data_path}/names.json", "r", encoding="utf-8") as f:
            self.names_map = json.load(f)

        with open(f"{kag_data_path}/kag_db.json", "r", encoding="utf-8") as f:
            self.kag_db = json.load(f)

    def build_context(self, all_lines: list[str], line_num: int) -> str:
        """
        Build full translation context for a given line.

        Args:
            all_lines (list[str]): All lines of the script.
            line_num (int): Index of the current line in `all_lines`.

        Returns:
            str: Multi-section context text including scene context, names dictionary,
                 and description of the speaking character or narration.
        """
        search_line = all_lines[line_num]
        prev_line = all_lines[line_num - 1].strip() if line_num > 0 else ""

        # scene context
        text_around = self._build_context_text_around(all_lines, line_num)
        # names dictionary
        names_dictionary = self._build_context_names_dict(search_line)
        # description of the speaking character or narration
        kag_description = self._build_context_kag_description(prev_line)

        if not text_around:
            return ""

        return (
            f"\n{text_around}"
            f"{names_dictionary}"
            f"\nОписание говорящего: {kag_description}\n"
        )

    def _build_context_text_around(self, all_lines: list[str], line_num: int) -> str:
        """
        Extracts scene context around a given line.
        """
        start = max(0, line_num - self.window_size)
        end = min(len(all_lines), line_num + self.window_size + 1)
        result = all_lines[start:end]
        return "Контекст сцены:\n" + "".join(result)

    def _build_context_kag_description(self, prev_line: str) -> str:
        """
        Determines description of the speaker or narration based on previous line.
        """
        if prev_line in self.kag_db:
            person = self.kag_db[prev_line]
            return f'прямая речь, {person["description"]}, пол: {person["gender"]}'
        return "не прямая речь, описание мира вокруг или действий персонажей"

    def _build_context_names_dict(self, search_line: str) -> str:
        """
        Collects a dictionary of names appearing in the current line.
        """
        relevant_names = {
            jp_name: ru_name
            for jp_name, ru_name in self.names_map.items()
            if jp_name in search_line
        }

        if not relevant_names:
            return ""

        names_block = "\nСловарь имён:\n" + "\n".join(
            f"{jp} = {ru}" for jp, ru in relevant_names.items()
        )
        return f"{names_block}\n"
