# -*- coding: utf-8 -*-
"""
@desc: RAG各环节评估脚本（已适配分层检索架构）
"""
import sys
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset

# --- 路径处理 ---
# 将项目根目录添加到Python路径中，以便能够导入agentic_rag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 评估所需的库 ---
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# --- 从项目中导入所需模块 ---
from agentic_rag.state import AgentState
from agentic_rag.nodes import (
    route_query_node, 
    generate_response_node, 
    rewrite_query_node,
    grade_relevance_node
)
# 导入新的分层检索器
from agentic_rag.hierarchical_retriever import hierarchical_retriever
# 导入项目中已配置好的llm和embedding function，用于传递给Ragas
from agentic_rag.chains import llm, get_embedding_function

# --- 全局配置 ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.csv")


def evaluate_router():
    """
    评估路由节点（route_query_node）的性能。
    此函数无需修改，因为route_query_node内部已更新为分层检索，改动被良好地封装了。
    """
    print("--- 开始评估【路由】环节 ---")
    
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"错误：评估数据集 '{DATASET_PATH}' 未找到ảng")
        return

    predictions = []
    ground_truth = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="测试路由节点"):
        question = row['question']
        ideal_route = row['ideal_route']
        
        initial_state = AgentState(query=question, documents=[], response="", route="", is_relevant=False, updated_query="", error=None)
        
        try:
            output_state = route_query_node(initial_state)
            predicted_route = output_state.get("route")
        except Exception as e:
            print(f"路由节点执行出错: {e}")
            predicted_route = "error"

        predictions.append(predicted_route)
        ground_truth.append(ideal_route)

    if not predictions:
        print("未能获取任何预测结果，请检查路由节点实现ảng")
        return
        
    print("\n--- 路由环节评估报告 ---")
    report = classification_report(ground_truth, predictions, zero_division=0)
    print(report)

    print("--- 混淆矩阵 ---")
    try:
        labels = sorted(list(set(ground_truth + predictions)))
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Router Performance Confusion Matrix")
        plt.savefig(os.path.join(os.path.dirname(__file__), "router_confusion_matrix.png"))
        print("混淆矩阵图已保存为 'router_confusion_matrix.png'")
    except Exception as e:
        print(f"无法生成混淆矩阵图: {e}")


def evaluate_generator_and_retriever():
    """
    评估检索器和生成器（retriever & generate_response_node）的性能。
    """
    print("\n--- 开始评估【检索和生成】环节 ---")
    
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"错误：评估数据集 '{DATASET_PATH}' 未找到ảng")
        return

    rag_questions_df = df[df['ideal_route'] == 'vectorstore'].copy()
    if rag_questions_df.empty:
        print("数据集中没有找到 'ideal_route' 为 'vectorstore' 的问题，跳过生成评估ảng")
        return

    questions = []
    generated_answers = []
    retrieved_contexts = []
    ideal_answers = []

    for index, row in tqdm(rag_questions_df.iterrows(), total=rag_questions_df.shape[0], desc="测试生成节点"):
        question = row['question']
        
        current_state = AgentState(query=question, documents=[], response="", route="vectorstore", is_relevant=False, updated_query="", error=None)

        try:
            rewrite_output = rewrite_query_node(current_state)
            current_state['updated_query'] = rewrite_output['updated_query']
        except Exception as e:
            print(f"重写节点执行出错: {e}")
            current_state['updated_query'] = question

        # --- 修改点：调用新的分层检索器 ---
        try:
            retrieved_docs = hierarchical_retriever(current_state['updated_query'])
            contexts = [doc.page_content for doc in retrieved_docs]
            current_state['documents'] = retrieved_docs
        except Exception as e:
            print(f"分层检索器执行出错: {e}")
            contexts = []
            current_state['documents'] = []

        try:
            generate_output = generate_response_node(current_state)
            answer = generate_output.get("response")
        except Exception as e:
            print(f"生成节点执行出错: {e}")
            answer = ""

        questions.append(question)
        generated_answers.append(answer)
        retrieved_contexts.append(contexts)
        ideal_answers.append(row['ideal_answer_summary'])

    if not questions:
        print("未能成功生成任何答案，无法进行Ragas评估ảng")
        return

    dataset_dict = {
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ideal_answers
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print("\n--- Ragas 评估报告 ---")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,       
                answer_relevancy,   
                context_recall,     
            ],
            llm=llm,
            embeddings=get_embedding_function(),
        )

        print(result)
        result_df = result.to_pandas()
        result_df.to_csv(os.path.join(os.path.dirname(__file__), "generator_ragas_report.csv"), index=False)
        print("Ragas评估报告已保存为 'generator_ragas_report.csv'")
    except Exception as e:
        print(f"Ragas评估执行出错: {e}")


def evaluate_grader():
    """
    评估相关性评估节点（grade_relevance_node）的性能。
    """
    print("\n--- 【评估】环节评估（占位） ---")
    print("评估'评估节点'需要一个专门的数据集ảng")


def main():
    """主函数，按顺序执行所有评估。"""
    evaluate_router()
    evaluate_generator_and_retriever()
    evaluate_grader()

if __name__ == "__main__":
    main()