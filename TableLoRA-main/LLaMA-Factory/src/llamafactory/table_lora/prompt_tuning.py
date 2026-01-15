import pandas as pd
from .table_lora import TABLE_TOKEN

def prompt_tuning_table_prompt(table: pd.DataFrame) -> str:
    """
    将 pandas DataFrame 转换为 TableLoRA 所需的特殊字符串格式。
    格式: Header <ROW> Cell1 <COL> Cell2 ...
    """
    # 1. 处理表头
    header_str = " | ".join([str(c) for c in table.columns])
    
    # 2. 处理每一行
    row_strs = []
    for _, row in table.iterrows():
        # 将行内单元格转为字符串并用 <COL> 连接
        # 注意：首个单元格前不需要 <COL>，因为它前面会加上 <ROW>
        row_cells = [str(cell) for cell in row]
        row_content = f" {TABLE_TOKEN.COL.value} ".join(row_cells)
        
        # 在行首添加 <ROW>
        row_strs.append(f"{TABLE_TOKEN.ROW.value} {row_content}")
        
    # 3. 拼接所有内容
    return header_str + " " + " ".join(row_strs)

# 导出 TABLE_TOKEN 以便 wikitq.py 可以从这里导入
__all__ = ["prompt_tuning_table_prompt", "TABLE_TOKEN"]