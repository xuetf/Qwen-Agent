import logging
import os
from typing import Literal, Optional, Dict, Union


class TruncatingFormatter(logging.Formatter):
    """截断过长日志内容的格式化器"""
    
    def __init__(self, fmt=None, datefmt=None, max_length=200):
        super().__init__(fmt, datefmt)
        self.max_length = max_length
    
    def format(self, record):
        # 先获取标准格式化的消息
        formatted_message = super().format(record)
        
        # 如果消息长度超过最大限制，则截断
        if len(formatted_message) > self.max_length:
            # 保留前后部分，中间用"..."替代
            prefix_length = self.max_length // 2 - 2
            suffix_length = self.max_length - prefix_length - 3
            truncated_message = formatted_message[:prefix_length] + "..." + formatted_message[-suffix_length:]
            return truncated_message
        
        return formatted_message


def setup_logger(
    level=None,
    output: Literal["terminal", "file", "both"] = "terminal",
    log_file: Optional[str] = None,
    concise: Union[bool, Dict[str, bool]] = False,
    max_length: Union[int, Dict[str, int]] = 200
):
    """
    设置日志记录器，支持终端输出、文件输出或同时输出到两者
    
    Args:
        level: 日志级别，默认根据环境变量QWEN_AGENT_DEBUG确定
        output: 输出目标，可选"terminal"、"file"或"both"
        log_file: 日志文件路径，当output为"file"或"both"时必须提供
        concise: 是否使用简约模式，可以是布尔值或字典{"terminal": bool, "file": bool}
        max_length: 简约模式下日志的最大长度，可以是整数或字典{"terminal": int, "file": int}
    """
    if level is None:
        if os.getenv('QWEN_AGENT_DEBUG', '0').strip().lower() in ('1', 'true'):
            level = logging.DEBUG
        else:
            level = logging.INFO
    
    # 处理简约模式配置
    if isinstance(concise, bool):
        concise_terminal = concise_file = concise
    else:
        concise_terminal = concise.get("terminal", False)
        concise_file = concise.get("file", False)
    
    # 处理最大长度配置
    if isinstance(max_length, int):
        max_length_terminal = max_length_file = max_length
    else:
        max_length_terminal = max_length.get("terminal", 200)
        max_length_file = max_length.get("file", 200)
    
    # 标准格式
    standard_format = '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
    # 简约格式
    concise_format = '%(levelname)s: %(message)s'
    
    _logger = logging.getLogger('qwen_agent_logger')
    _logger.setLevel(level)
    # 清除已有的处理器，以避免重复添加
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)
    
    if output in ["terminal", "both"]:
        # 终端输出处理器
        console_handler = logging.StreamHandler()
        
        # 根据简约模式选择格式
        if concise_terminal:
            formatter = TruncatingFormatter(concise_format, max_length=max_length_terminal)
        else:
            formatter = logging.Formatter(standard_format)
            
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)
    
    if output in ["file", "both"]:
        if log_file is None:
            # 如果没有提供日志文件路径，则使用默认路径
            log_dir = os.path.join(os.path.expanduser("~"), ".qwen_agent_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "qwen_agent.log")
        
        # 确保日志文件的目录存在
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        # 文件输出处理器
        file_handler = logging.FileHandler(log_file)
        
        # 根据简约模式选择格式
        if concise_file:
            formatter = TruncatingFormatter(concise_format, max_length=max_length_file)
        else:
            formatter = logging.Formatter(standard_format)
            
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    
    return _logger


os.environ['QWEN_AGENT_DEBUG'] = '1'
logger = setup_logger(output='both', concise={'terminal': True, 'file': False},
                      log_file='qwen_agent.log', max_length=500)

# 提供一个API以便在运行时重新配置日志器
def reconfigure_logger(
    level=None, 
    output="terminal", 
    log_file=None, 
    concise: Union[bool, Dict[str, bool]] = False,
    max_length: Union[int, Dict[str, int]] = 200
):
    """
    重新配置日志记录器
    
    Args:
        level: 日志级别
        output: 输出目标，可选"terminal"、"file"或"both"
        log_file: 日志文件路径，当output为"file"或"both"时使用
        concise: 是否使用简约模式，可以是布尔值或字典{"terminal": bool, "file": bool}
        max_length: 简约模式下日志的最大长度，可以是整数或字典{"terminal": int, "file": int}
    """
    global logger
    logger = setup_logger(
        level=level, 
        output=output, 
        log_file=log_file, 
        concise=concise, 
        max_length=max_length
    )
    return logger
