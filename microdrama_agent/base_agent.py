import json
import os

class BaseAgent:
    def __init__(self, llm_type, system_prompt="", use_history=True, temp=0, top_p=1):
        self.use_history = use_history
        self.llm_type = llm_type
        self.streaming = False
        # Lazy-init OpenAI client.
        # This allows running pipelines that reuse existing Step_*.json without requiring
        # the `openai` package (as long as no LLM calls are made).
        self.client = None
        if llm_type in ("deepseek-r1", "deepseek-v3"):
            self.streaming = True
        
        self.system = system_prompt
        self.temp = temp
        self.top_p = top_p
        self.input_tokens_count = 0
        self.output_tokens_count = 0
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system_prompt})

    def _ensure_client(self):
        if self.client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ModuleNotFoundError(
                "Missing dependency: openai. Install it (e.g. `pip install openai`) or run stages that don't need LLM. "
                f"Original error: {e}"
            )

        if self.llm_type == "gpt4-o":
            self.client = OpenAI()
        elif self.llm_type in ("deepseek-r1", "deepseek-v3"):
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
    
    
    def __call__(self, message, parse=False):
        self.messages.append({"role": "user", "content": message})
        result = self.generate(message, parse)
        self.messages.append({"role": "assistant", "content": result})

        print(result)
        if parse:
            try:
                result = self.parse_json(result)
            except:
                raise Exception("Error content is list below:\n", result)
            
        return result
        
    
    
    def generate(self, message, json_format):
        self._ensure_client()
        if self.use_history:
            input_messages = self.messages
        else:
            input_messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": message}
            ]
            
        
        if self.llm_type == "gpt4-o":
            if json_format:
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-08-06", # gpt-4
                    messages=input_messages,
                    temperature=self.temp,
                    top_p=self.top_p,
                    response_format = { "type": "json_object" }
                    )
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-08-06", # gpt-4
                    messages=input_messages,
                    temperature=self.temp,
                    top_p=self.top_p,
                    )
        elif self.llm_type == "deepseek-r1":
            
            if not self.streaming:
                response = self.client.chat.completions.create(
                                model="deepseek-r1",  # deepseek-r1
                                messages=input_messages
                            )
            else:
                # 流式回复
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""     # 定义完整回复
                is_answering = False   # 判断是否结束思考过程并开始回复

                completion = self.client.chat.completions.create(
                    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称
                    messages=input_messages,
                    stream=True
                )
                for chunk in completion:
                    # 如果chunk.choices为空，则打印usage
                    if not chunk.choices:
                        print("\nUsage:")
                        print(chunk.usage)
                    else:
                        delta = chunk.choices[0].delta
                        # 打印思考过程
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            print(delta.reasoning_content, end='', flush=True)
                            reasoning_content += delta.reasoning_content
                        else:
                            # 开始回复
                            if delta.content != "" and is_answering == False:
                                is_answering = True
                            # 打印回复过程
                            print(delta.content, end='', flush=True)
                            answer_content += delta.content
                return answer_content
        elif self.llm_type == "deepseek-v3":
            
            if not self.streaming:
                response = self.client.chat.completions.create(
                                model="deepseek-v3", 
                                messages=input_messages
                            )
            else:
                # 流式回复
                reasoning_content = ""  # 定义完整思考过程
                answer_content = ""     # 定义完整回复
                is_answering = False   # 判断是否结束思考过程并开始回复

                completion = self.client.chat.completions.create(
                    model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称
                    messages=input_messages,
                    stream=True
                )
                for chunk in completion:
                    # 如果chunk.choices为空，则打印usage
                    if not chunk.choices:
                        print("\nUsage:")
                        print(chunk.usage)
                    else:
                        delta = chunk.choices[0].delta
                        # 打印思考过程
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            print(delta.reasoning_content, end='', flush=True)
                            reasoning_content += delta.reasoning_content
                        else:
                            # 开始回复
                            if delta.content != "" and is_answering == False:
                                is_answering = True
                            # 打印回复过程
                            print(delta.content, end='', flush=True)
                            answer_content += delta.content
                return answer_content
            
        else:
            response = self.client.chat.completions.create(
                            model=self.llm_type,  
                            messages=input_messages
                        )
            
        self.update_tokens_count(response)
        return response.choices[0].message.content
    
    
    def parse_json(self, response):
        response = response.replace("```json","")
        response = response.replace("```","")
        return json.loads(response)

    
    def add(self, message: dict):
        self.messages.append(message)
    
    
    def update_tokens_count(self, response):
        self.input_tokens_count += response.usage.prompt_tokens
        self.output_tokens_count += response.usage.completion_tokens
    
    
    def show_usage(self):
        print(f"Total input tokens used: {self.input_tokens_count}\nTotal output tokens used: {self.output_tokens_count}")
        



