from src.file_workflow import FileWorkflow
from src.llm_workflow import LlmWorkflow
from config import ROOT_DIR


FILE_NAME = "123"


llm = LlmWorkflow("IlyaGusev/saiga_gemma3_12b")
file_workflow = FileWorkflow(llm, show_prompts=False, kag_enabled=True)


file_workflow.translate_one_file(
    input_path=f"{ROOT_DIR}/input/{FILE_NAME}",
    output_path=f"{ROOT_DIR}/output/{FILE_NAME}",
    encoding="UTF-8"
)
