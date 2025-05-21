import copy
import json
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Tuple, Union

import openai

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create
    
    def _has_tool_calls(self, obj, attribute='tool_calls') -> bool:
        """
        检查对象是否包含工具调用并且调用列表非空
        
        Args:
            obj: 要检查的对象
            attribute: 工具调用的属性名称
            
        Returns:
            bool: 如果存在非空的工具调用返回True，否则返回False
        """
        return hasattr(obj, attribute) and getattr(obj, attribute)
            
    def _format_tool_call(self, tool_call = None, tool_call_data: Dict = None) -> Tuple[str, bool]:
        """
        将工具调用格式化为特定格式的字符串
        
        Args:
            tool_call: OpenAI返回的工具调用对象
            tool_call_data: 包含name和arguments的字典对象
            
        Returns:
            Tuple[str, bool]: 格式化后的字符串和是否成功格式化的标志
        """
        function_name = None
        arguments = None
        
        # 从tool_call对象中提取函数名和参数
        if tool_call is not None and hasattr(tool_call, 'function') and tool_call.function:
            function_name = getattr(tool_call.function, 'name', '')
            arguments = getattr(tool_call.function, 'arguments', '')
        # 从tool_call_data字典中提取函数名和参数
        elif tool_call_data is not None:
            function_name = tool_call_data.get('name', '')
            arguments = tool_call_data.get('arguments', '')
        
        if function_name is None or arguments is None:
            return "", False
            
        function_name = function_name or ""
        arguments_str = arguments if arguments else '{}'
        
        try:
            # 尝试解析arguments为JSON对象
            args_obj = json.loads(arguments_str)
            tool_call_obj = {"name": function_name, "arguments": args_obj}
            tool_call_json = json.dumps(tool_call_obj, ensure_ascii=False)
            return f"\n<tool_call>\n{tool_call_json}\n</tool_call>", True
        except json.JSONDecodeError:
            # 如果解析失败，使用空对象
            tool_call_obj = {"name": function_name, "arguments": {}}
            tool_call_json = json.dumps(tool_call_obj, ensure_ascii=False)
            return f"\n<tool_call>\n{tool_call_json}\n</tool_call>", True

    def _process_tool_call_chunk(self, tool_call, full_tool_calls: List[Dict]) -> None:
        """
        处理工具调用的数据块，收集索引和参数
        
        Args:
            tool_call: OpenAI返回的工具调用对象
            full_tool_calls: 用于收集工具调用信息的列表
        """
        # 寻找是否已有相同id的tool_call
        if hasattr(tool_call, 'index'):
            idx = tool_call.index
            # 确保数组大小足够
            while len(full_tool_calls) <= idx:
                full_tool_calls.append({'name': '', 'arguments': ''})
            
            # 追加function name和arguments
            if hasattr(tool_call, 'function'):
                if hasattr(tool_call.function, 'name'):
                    name = tool_call.function.name
                    if name is not None:
                        full_tool_calls[idx]['name'] = name
                if hasattr(tool_call.function, 'arguments'):
                    arguments = tool_call.function.arguments
                    if arguments is not None:
                        full_tool_calls[idx]['arguments'] += arguments

    def _should_process_tool_calls(self) -> bool:
        """
        判断是否应该处理工具调用
        
        Returns:
            bool: 如果应该处理工具调用返回True，否则返回False
        """
        # 检查generate_cfg中保存的原始fncall_prompt_type，或检查self.fncall_prompt的类型
        return hasattr(self, 'fncall_prompt') and self.fncall_prompt.__class__.__name__ == 'NousFnCallPrompt'

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        process_tool_calls = self._should_process_tool_calls()
        
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta,
                                   'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=chunk.choices[0].delta.reasoning_content)
                            ]
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
                        
                        # 处理tool_calls字段
                        if process_tool_calls and self._has_tool_calls(chunk.choices[0].delta):
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                # 格式化工具调用
                                tool_call_str, success = self._format_tool_call(tool_call=tool_call)
                                if success:
                                    yield [Message(role=ASSISTANT, content=tool_call_str)]
            else:
                full_response = ''
                full_reasoning_content = ''
                full_tool_calls = []
                
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta,
                                   'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            full_reasoning_content += chunk.choices[0].delta.reasoning_content
                        
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        
                        # 收集tool_calls
                        if process_tool_calls and self._has_tool_calls(chunk.choices[0].delta):
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                self._process_tool_call_chunk(tool_call, full_tool_calls)
                        
                        # 构建完整响应
                        combined_response = full_response
                        
                        # 添加已完成的tool_calls
                        if process_tool_calls:
                            for tc in full_tool_calls:
                                tool_call_str, success = self._format_tool_call(tool_call_data=tc)
                                if success:
                                    combined_response += tool_call_str
                        
                        yield [Message(role=ASSISTANT, content=combined_response, reasoning_content=full_reasoning_content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        process_tool_calls = self._should_process_tool_calls()
        
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            content = response.choices[0].message.content or ""
            
            # 处理tool_calls
            if process_tool_calls and self._has_tool_calls(response.choices[0].message):
                for tool_call in response.choices[0].message.tool_calls:
                    tool_call_str, success = self._format_tool_call(tool_call=tool_call)
                    if success:
                        content += tool_call_str
            
            if hasattr(response.choices[0].message, 'reasoning_content'):
                return [
                    Message(role=ASSISTANT,
                            content=content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            return [Message(role=ASSISTANT, content=content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [msg.model_dump() for msg in messages]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input After process: \n{pformat(messages, indent=2)}')
        return messages
