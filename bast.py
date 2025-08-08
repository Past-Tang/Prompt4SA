import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置SWIFT环境变量来抑制WARNING日志
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['LOG_LEVEL'] = 'ERROR'  # 抑制SWIFT的WARNING信息
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import tqdm
from matplotlib import rcParams
from PIL import Image
from swift.llm import InferRequest, VllmEngine, RequestConfig

# --- 常量配置 ---
MODEL_PATH = ''
VQA_DATA_PATH = 'data/VQA-SA-question.json'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'InternVL3-output')
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), 'VQA-SA-results.json')
FONT_PATH = os.path.join(os.path.dirname(__file__), 'src/NotoSansSC-Regular.otf')
SUBMIT_TO_EVALAI = True  # 全局开关：设置为 True 以启用提交功能

# --- 可视化多线程配置 ---
ENABLE_MULTITHREADED_VISUALIZATION = True  # 启用多线程可视化加速
MAX_VISUALIZATION_WORKERS = None  # 最大线程数，None表示使用CPU核心数

# --- 主逻辑 ---

def main():
    """
    视觉问答批量推理主程序。
    """
    # 1. 初始化环境
    setup_matplotlib_font()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已创建/确认存在: {OUTPUT_DIR}")

    # 2. 加载数据
    vqa_data = load_vqa_data(VQA_DATA_PATH)
    if not vqa_data:
        print("未能加载VQA数据，程序终止。")
        return
    # 单进程模式: 使用所有配置的GPU
    # 3. 初始化推理引擎
    engine = initialize_inference_engine(MODEL_PATH)
    request_config = create_request_config()

    # 4. 准备推理请求
    print("开始准备推理请求...")
    requests, items = prepare_inference_requests(vqa_data)
    if not requests:
        print("没有可处理的有效推理请求，程序终止。")
        return

    # 5. 执行批量推理
    print("开始执行批量推理...")
    responses = engine.infer(requests, request_config,use_tqdm=True)
    print("批量推理完成。")

    # 6. 处理推理响应
    print("开始处理推理响应...")
    results = process_responses(responses, items)
    print("响应处理完成。")

    # 7. 生成并保存可视化结果
    print("开始生成可视化结果...")
    if ENABLE_MULTITHREADED_VISUALIZATION:
        generate_visualizations(results, max_workers=MAX_VISUALIZATION_WORKERS)
    else:
        generate_visualizations_sequential(results)
    print("可视化结果生成完成。")

    # 8. 保存最终的聚合结果
    save_aggregated_results(results, RESULTS_JSON_PATH)

    print("\n所有任务已完成。")
    print(f"聚合结果已保存至: {RESULTS_JSON_PATH}")
    print(f"可视化结果图像已保存至: {OUTPUT_DIR}")

    # 9. 提交到 EvalAI
    if SUBMIT_TO_EVALAI:
        CHALLENGE_ID = "2552"
        PHASE_ID = "5069"  # 例如: "5557"
        SUBMISSION_TYPE = "--public"      # 可选项: "--public" 或 "--private"

        # 可选: 预设提交细节，如果不想填写，请保持值为空字符串 ""
        SUBMISSION_DETAILS = {
            "method_name": "",
            "method_description": "",
            "project_url": "",
            "publication_url": ""
        }

        if PHASE_ID and PHASE_ID != "YOUR_PHASE_ID_HERE":
            submit_to_evalai(
                file_path=RESULTS_JSON_PATH,
                challenge_id=CHALLENGE_ID,
                phase_id=PHASE_ID,
                submission_type=SUBMISSION_TYPE,
                details=SUBMISSION_DETAILS
            )
        else:
            print("\n" + "="*50)
            print("提醒: 已跳过向 EvalAI 提交结果的步骤。")
            print(f"原因: `PHASE_ID` 未设置或无效。请在脚本 `bast.py` 中设置有效的 `PHASE_ID`。")
            print("="*50)
    else:
        print("\n" + "="*50)
        print("提醒: 根据全局设置，已跳过向 EvalAI 提交结果的步骤。")
        print(f"如需自动提交，请在脚本 `bast.py` 中将 `SUBMIT_TO_EVALAI` 设置为 `True`。")
        print("="*50)

def process_data_part(data_part, gpu_ids):
    """处理数据的一部分，使用指定的GPU"""
    
    try:
        print(f"使用GPU {gpu_ids} 处理 {len(data_part)} 条数据...")
        
        # 初始化推理引擎(这里只使用4个GPU，所以tensor_parallel_size保持为4)
        engine = initialize_inference_engine(MODEL_PATH)
        request_config = create_request_config()
        
        # 准备推理请求
        requests, items = prepare_inference_requests(data_part)
        if requests:
            # 执行批量推理
            responses = engine.infer(requests, request_config, use_tqdm=True)
            
            # 处理推理响应
            results = process_responses(responses, items)
            
            # 生成可视化结果
            if ENABLE_MULTITHREADED_VISUALIZATION:
                generate_visualizations(results, max_workers=MAX_VISUALIZATION_WORKERS)
            else:
                generate_visualizations_sequential(results)
            
            # 保存部分结果
            part_results_path = os.path.join(os.path.dirname(RESULTS_JSON_PATH), 
                                            f"part_{gpu_ids.replace(',', '_')}_results.json")
            save_aggregated_results(results, part_results_path)
            
            return results
        else:
            print(f"GPU {gpu_ids} 没有可处理的有效推理请求。")
            return []
    finally:
        pass

# --- 核心功能函数 ---

def load_vqa_data(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载VQA数据。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"正在从 {file_path} 加载数据...")
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到数据文件 {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式。")
        return []

def initialize_inference_engine(model_path: str) -> VllmEngine:
    """初始化并返回SWIFT推理引擎。"""
    print(f"正在加载模型: {model_path}...")
    engine = VllmEngine(model_path,
                        max_model_len=32768,
                        tensor_parallel_size=4,
                        gpu_memory_utilization=0.95,  # 提高GPU内存利用率
                        max_num_seqs=32,  # 最大并行序列数
                     
                        )
    return engine

def create_request_config() -> RequestConfig:
    """创建并返回推理请求配置。"""
    return RequestConfig(
        max_tokens=32768,
        temperature=0.1,
        top_p=0.8,
        repetition_penalty=1.05,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

def prepare_inference_requests(data: List[Dict[str, Any]]) -> Tuple[List[InferRequest], List[Dict[str, Any]]]:
    """
    根据输入数据准备推理请求。

    返回:
        - infer_requests: 用于模型推理的请求列表。
        - items_for_requests: 与请求列表对应的原始数据项。
    """
    infer_requests = []
    items_for_requests = []
    base_dir = os.path.dirname(__file__)

    for item in tqdm.tqdm(data, desc="准备推理请求"):
        question = item.get("question")
        image_path = item.get("image_path")

        if not question or not image_path:
            continue

        full_image_path = os.path.join(base_dir, image_path)
        if not os.path.exists(full_image_path):
            print(f"警告: 图片未找到，已跳过 -> {full_image_path}")
            continue
        
        # 提取历史记录
        history = item.get("history")

        messages = build_messages(question, history)
        req = InferRequest(messages=messages, images=[full_image_path])
        infer_requests.append(req)
        items_for_requests.append(item)
    
    return infer_requests, items_for_requests

def process_responses(responses: List[Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理模型的推理响应。"""
    results = []
    for resp, item in tqdm.tqdm(zip(responses, items), total=len(responses), desc="处理响应"):
        question = item.get("question")
        image_path = item.get("image_path")
        history = item.get("history")

        try:
            content = resp.choices[0].message.content
            answer = clean_content(content)
        except (KeyError, IndexError, AttributeError):
            print(f"无法从响应中提取有效答案: {resp}")
            answer = "无回答"

        result_item = {
            "image_path": image_path,
            "question": question,
            "result": answer,
        }
        if history:
            result_item["history"] = history
            
        results.append(result_item)
        
        # 打印当前处理结果
        print(f"问题: {question}")
        print(f"回答: {answer}")
        print("-" * 50)
    
    return results

def generate_visualizations(results: List[Dict[str, Any]], max_workers: int = None):
    """
    根据处理后的结果生成并保存可视化图像。
    每个图像文件对应一张包含其所有问答对的聚合图。
    使用多线程加速处理。

    Args:
        results: 处理后的结果列表
        max_workers: 最大线程数，默认为CPU核心数
    """
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[res['image_path']].append(res)

    # 当无可用结果时安全退出
    if not grouped_results:
        print("没有可用于可视化的结果，已跳过图像生成。")
        return

    # 设置线程数，默认为CPU核心数，但不超过图像数量
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(grouped_results))
    # 保证线程数为正
    if max_workers is not None and max_workers < 1:
        max_workers = 1

    print(f"使用 {max_workers} 个线程并行生成可视化图像...")

    # 准备任务列表
    tasks = []
    for image_path, qa_pairs in grouped_results.items():
        image_basename = os.path.basename(image_path)
        output_filename_base = os.path.splitext(image_basename)[0]
        output_filepath = os.path.join(OUTPUT_DIR, f"{output_filename_base}.png")
        tasks.append((image_path, qa_pairs, output_filepath))

    # 使用线程池执行任务
    successful_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_visualization, task): task
            for task in tasks
        }

        # 使用tqdm显示进度
        with tqdm.tqdm(total=len(tasks), desc="生成可视化图像") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                image_path = task[0]
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                    successful_count += 1
                except Exception as e:
                    print(f"处理图像 {image_path} 时发生错误: {e}")
                    failed_count += 1
                finally:
                    pbar.update(1)

    print(f"可视化生成完成: 成功 {successful_count} 个，失败 {failed_count} 个")


def generate_visualizations_sequential(results: List[Dict[str, Any]]):
    """
    顺序生成可视化图像（原始版本，作为备选方案）。
    """
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[res['image_path']].append(res)

    for image_path, qa_pairs in tqdm.tqdm(grouped_results.items(), desc="生成可视化图像（顺序）"):
        try:
            image_basename = os.path.basename(image_path)
            output_filename_base = os.path.splitext(image_basename)[0]
            output_filepath = os.path.join(OUTPUT_DIR, f"{output_filename_base}.png")
            save_result_image(image_path, qa_pairs, output_filepath)
        except Exception as e:
            print(f"处理图像 {image_path} 时发生错误: {e}")


def process_single_visualization(task_data: Tuple[str, List[Dict[str, Any]], str]):
    """
    处理单个图像的可视化任务。
    这个函数被设计为线程安全的。

    Args:
        task_data: 包含 (image_path, qa_pairs, output_filepath) 的元组
    """
    image_path, qa_pairs, output_filepath = task_data

    # 为每个线程设置matplotlib后端，确保线程安全
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端

    try:
        save_result_image(image_path, qa_pairs, output_filepath)
    except Exception as e:
        # 重新抛出异常，让调用者处理
        raise Exception(f"处理图像 {image_path} 失败: {str(e)}")

def save_aggregated_results(results: List[Dict[str, Any]], file_path: str):
    """将所有结果聚合保存到单个JSON文件，并调整路径格式。"""
    results_to_save = []
    for item in results:
        new_item = item.copy()
        if 'image_path' in new_item and isinstance(new_item['image_path'], str):
            new_item['image_path'] = new_item['image_path'].replace('/', '\\')
        results_to_save.append(new_item)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)


def submit_to_evalai(file_path: str, challenge_id: str, phase_id: str, submission_type: str, details: Dict[str, str] = None):
    """使用 evalai-cli 提交结果文件，并自动处理交互式提示。"""
    import subprocess

    cmd = [
        "evalai", "challenge", challenge_id, "phase", phase_id, "submit",
        "--file", file_path,
        "--large"
    ]

    if submission_type in ["--public", "--private"]:
        cmd.append(submission_type)
    else:
        print(f"警告: 无效的提交类型 '{submission_type}'。将使用 EvalAI 的默认设置。")

    # 根据是否提供了 details，构建用于自动应答的输入字符串
    if details and any(details.values()):
        # 回答 "y"，然后依次提供所有细节，最后用换行符结束
        input_string = (
            "y\n"
            f"{details.get('method_name', '')}\n"
            f"{details.get('method_description', '')}\n"
            f"{details.get('project_url', '')}\n"
            f"{details.get('publication_url', '')}\n"
        )
        print("\n将使用预设的提交细节自动填充。")
    else:
        # 如果没有提供任何细节，则回答 "n" 以跳过交互
        input_string = "n\n"
        print("\n未提供提交细节，将跳过交互式提问。")


    print("\n" + "="*50)
    print("准备提交到 EvalAI...")
    print(f"执行命令: {' '.join(cmd)}")
    print("="*50)

    try:
        # Pass the constructed input string to stdin to automate prompts
        process = subprocess.run(
            cmd,
            input=input_string,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print("命令输出:")
        print(process.stdout)
        print("\n文件提交成功！")
    except FileNotFoundError:
        print("\n错误: 'evalai' 命令未找到。")
        print("请确保 evalai-cli 已安装并位于系统的 PATH 环境变量中。")
        print("您可以通过 `pip install evalai` 进行安装。")
    except subprocess.CalledProcessError as e:
        print("\n错误: 文件提交失败。")
        print("返回码:", e.returncode)
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

# --- 辅助工具函数 ---

def setup_matplotlib_font():
    """设置matplotlib以支持中文字体显示。"""
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        sans_serif_fonts = [prop.get_name(), 'Microsoft YaHei', 'SimHei']
        print(f"已加载自定义字体: {prop.get_name()}")
    else:
        print(f"警告：未找到字体文件 {FONT_PATH}。将尝试使用系统默认字体。")
        sans_serif_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = sans_serif_fonts
    rcParams['axes.unicode_minus'] = False

def clean_content(content: str) -> str:
    """清理模型输出内容，移除特殊标记。"""
    if not content:
        return "无回答"
    content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    import re
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, content, flags=re.S)   # re.S 让 . 匹配换行
    if match:
        result = match.group(1)
        return str(result).strip()
    else:
         return str(content).strip()
   

def build_messages(question: str, history: List[List[str]] = None) -> List[Dict[str, str]]:
    """构建发送给模型的聊天消息, 支持多轮对话。"""
    
    tqtext=r'''并将最终答案使用<answer></answer>包裹'''
    
    # 基于VQA-SA数据集特点的认知科学导向prompt
    task_instruction = f'''
    你是一个具备高级空间智能的AI助手，专门解决视觉空间推理问题。请运用以下认知框架来处理空间关系问题：

    ## 🧠 认知处理框架

    ### 阶段1: 问题分析与分级 (Question Analysis & Classification)
    首先判断问题的认知复杂度：
    - **低复杂度**: 简单的对象识别或场景判断
    - **中复杂度**: 基础空间关系("A在B的哪个方向？")  
    - **高复杂度**: 视角转换("假设你是...从你的角度看...")
    - **极高复杂度**: 复合特征+视角转换+多向空间推理

    ### 阶段2: 特征绑定与对象识别 (Feature Binding & Object Recognition)
    精确识别问题中的关键实体：
    - **复合特征解析**: "穿白色衣服的人" = 颜色特征 + 衣着特征 + 人物类别
    - **动态状态理解**: "骑自行车的人" = 动作状态 + 人物类别 + 工具关系
    - **空间定位**: "坐在路边的人" = 位置状态 + 环境关系 + 人物类别
    
    ### 阶段3: 参考系统建立 (Reference Frame Construction)
    根据问题类型建立正确的空间参考系：
    
    **A. 客观参考系（71.2%的问题）**:
    - 以图像本身为参考
    - 前方=图像上方，后方=图像下方，左方=图像左侧，右方=图像右侧

    **B. 主观参考系（18.6%的问题）**:
    - "拍摄者视角": 以摄影师位置为原点
    - "图中人物视角": 需要识别人物朝向，建立以该人物为中心的坐标系

    **C. 第一人称参考系（最高难度）**:
    - "假设你是图中的X": 进行完整的视角转换
    - 步骤: 定位X → 判断X的朝向 → 建立X的前后左右坐标系 → 重新计算所有空间关系

    ### 阶段4: 空间关系计算 (Spatial Relationship Computation)
    系统化处理空间关系：
    
    **4.1 方位计算**:
    - 基础4向: 前、后、左、右
    - 复合8向: 左前、右前、左后、右后、正前、正后、正左、正右  
    - 立体6向: 加上"上方"、"下方"
    
    **4.2 距离估算**:
    - 近距离: "1米内" "紧挨着"
    - 中距离: "几米" "10米内"
    - 远距离: "远处" "背景中"
    
    **4.3 相对关系**:
    - 比较距离: "A比B离C更近"
    - 遮挡关系: "A在B的前面/后面"

    ### 阶段5: 认知验证与输出 (Cognitive Verification & Output)
    进行多层次验证：
    - **逻辑一致性**: 空间关系是否符合物理常识？
    - **视角准确性**: 是否正确进行了视角转换？
    - **答案格式**: 是否符合问题要求的输出格式？

    ## 🎯 特殊情况处理指南

    ### 视角转换问题的处理策略：
    1. **识别视角主体**: 准确定位"你"指代的对象
    2. **确定主体朝向**: 通过身体姿态、注视方向判断前方
    3. **重构坐标系**: 以主体为原点，其朝向为前方，建立新的方位体系
    4. **重新计算**: 在新坐标系下计算目标对象的相对位置

    ### 复合特征问题的处理策略：
    1. **特征分解**: 将"穿白色衣服戴红色帽子的人"分解为多个识别条件
    2. **逐步筛选**: 先找出所有穿白色衣服的人，再筛选戴红色帽子的
    3. **唯一确定**: 确保识别出唯一的目标对象

    ### 距离量化问题的处理策略：
    1. **比例估算**: 以人体高度（约1.7米）为参考
    2. **环境线索**: 利用道路宽度、建筑尺度等环境信息
    3. **深度感知**: 结合透视关系和遮挡线索

    请严格按照这个认知框架进行推理，确保每一步都准确无误。记住：空间推理需要精确性，一个小错误就会导致完全错误的结果。

    {tqtext}
    '''


    # 基础消息，包含系统角色
    messages = [
        {"role": "system", "content": """你是一个具备高级空间智能的AI助手，专门解决复杂的视觉空间推理问题。你的核心能力包括：

🧠 认知能力：
- 多层次空间表征：能够在2D图像中理解3D空间关系
- 视角转换能力：能够进行第一人称、第三人称视角的灵活切换
- 特征绑定能力：能够同时处理多个物体特征进行精确识别
- 心理旋转能力：能够在心理空间中旋转和操作空间模型

🎯 推理特长：
- 相对空间推理：擅长处理"A在B的哪个方向"类型问题
- 距离估算：能够基于视觉线索进行空间距离的准确判断
- 复合关系分析：能够处理多对象之间的复杂空间关系网络
- 动态场景理解：能够理解运动状态下的空间关系

⚡ 处理原则：
- 系统性思考：按照认知科学框架逐步分析
- 精确性导向：空间推理不允许模糊和错误
- 适应性强：根据问题复杂度采用不同的处理策略
- 验证机制：对推理结果进行多重检验

你将接收包含图像和空间关系问题的输入，并提供准确、简洁的空间推理答案。"""}]
    
    # 构建对话历史
    chat_history = []
    
    if history:
        for old_q, old_a in history:
            chat_history.append({"role": "user", "content": f"问题: `{old_q}`"})
            chat_history.append({"role": "assistant", "content": old_a})

    # 将当前问题作为最后一轮用户输入
    chat_history.append({"role": "user", "content": f"问题: `{question}`"})

    # 在第一轮用户消息前加上图片占位符和任务说明
    if chat_history:
        first_user_message = chat_history[0]['content']
        chat_history[0]['content'] = f"<|image|>\n{task_instruction}\n{first_user_message}"
    
    messages.extend(chat_history)
    
    return messages

def save_result_image(image_path: str, qa_pairs: List[Dict[str, Any]], output_filepath: str):
    """将单个图像及其所有相关的问答对保存为一张可视化图片。线程安全版本。"""

    # 确保matplotlib使用非交互式后端
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 创建新的figure，避免全局状态冲突
    fig, axs = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [1.2, 0.8]}, dpi=200)
    
    # 1. 显示图像
    full_image_path = os.path.join(os.path.dirname(__file__), image_path)
    try:
        img = Image.open(full_image_path)
        axs[0].imshow(img)
        axs[0].set_title(f"图像: {os.path.basename(image_path)}", fontsize=16)
    except Exception as e:
        axs[0].text(0.5, 0.5, f"图像加载失败: {e}", ha='center', va='center', color='red', fontsize=14)
        axs[0].set_title(f"图像路径: {image_path}", fontsize=16)
    finally:
        axs[0].axis('off')

    # 2. 显示所有问答文本
    axs[1].axis('off')
    y_pos = 0.98
    line_height = 0.035
    max_chars_per_line = 45

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        answer_text = qa["result"]

        if y_pos < 0.1: # 检查是否有足够空间
            axs[1].text(0.05, y_pos, "...", fontsize=16, color='gray')
            break

        # 显示问题
        axs[1].text(0.05, y_pos, f"问题 #{i+1}:", fontsize=18, fontweight='bold', color='blue')
        y_pos -= line_height
        question_lines = [question[j:j+max_chars_per_line] for j in range(0, len(question), max_chars_per_line)]
        for line in question_lines:
            axs[1].text(0.05, y_pos, line, fontsize=16, wrap=True)
            y_pos -= line_height

        # 显示回答
        y_pos -= line_height * 0.25 # 增加问题和回答之间的间距
        axs[1].text(0.05, y_pos, "回答:", fontsize=18, fontweight='bold', color='green')
        y_pos -= line_height
        answer_lines = [answer_text[j:j+max_chars_per_line] for j in range(0, len(answer_text), max_chars_per_line)]
        for line in answer_lines:
            axs[1].text(0.05, y_pos, line, fontsize=14, wrap=True)
            y_pos -= line_height

        # 在问答对之间添加分隔符
        if i < len(qa_pairs) - 1:
            y_pos -= line_height * 0.5
            axs[1].text(0.05, y_pos, "-" * 50, fontsize=14, color='gray')
            y_pos -= line_height
    
    fig.suptitle(f"视觉问答聚合结果: {os.path.basename(image_path)}", fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    fig.savefig(output_filepath, bbox_inches='tight')

    # 显式关闭figure释放内存
    plt.close(fig)

    # 使用线程锁保护打印输出（可选）
    print(f"已保存聚合结果图像到: {output_filepath}")


if __name__ == "__main__":
    main()
