# # ui/app.py
# import os
# import gradio as gr
# from llama_cpp import Llama

# # ---------- CONFIG ----------
# MODEL_PATH = r"C:\Users\222003150\project\FinancialAssistantAI\models\finance-Llama3-8B.Q2_K.gguf"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# # Instantiate the Llama model
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=2048,
#     n_threads=4,
#     n_batch=128,
#     verbose=False
# )

# # Default system prompt
# DEFAULT_SYSTEM_PROMPT = (
#     "You are FinWise, a professional financial assistant. "
#     "Provide clear, structured, and practical financial insights. "
#     "If uncertain, state it honestly instead of fabricating answers."
# )

# # ---------- RESPONSE LOGIC ----------
# def generate_response(user_query, query_type):
#     # Mock response for testing and fallback
#     mock_response = {
#         "answer": f"Here’s a brief insight about '{query_type}' for your input: '{user_query}'.",
#         "reasoning": "The AI considered financial ratios, current market indicators, and risk factors.",
#         "risk": "Moderate risk due to volatility in the financial sector."
#     }

#     try:
#         # Create a structured system prompt
#         system_prompt = (
#             f"{DEFAULT_SYSTEM_PROMPT}\n\n"
#             f"User Query Type: {query_type}\n"
#             f"User Question: {user_query}\n\n"
#             "Respond with structured financial insights, reasoning, and risk evaluation."
#         )

#         # Generate response from Llama model
#         raw_response = llm(
#             prompt=system_prompt,
#             max_tokens=512,
#             temperature=0.7,
#             top_p=0.9,
#             stop=["User:", "Assistant:"]
#         )

#         answer_text = raw_response["choices"][0]["text"].strip()

#         # Simple split simulation to separate reasoning/risk (for demo)
#         if "Reasoning:" in answer_text and "Risk:" in answer_text:
#             parts = answer_text.split("Reasoning:")
#             insight = parts[0].strip()
#             subparts = parts[1].split("Risk:")
#             reasoning = subparts[0].strip()
#             risk = subparts[1].strip() if len(subparts) > 1 else mock_response["risk"]
#             return insight, reasoning, risk
#         else:
#             return mock_response["answer"], mock_response["reasoning"], mock_response["risk"]

#     except Exception:
#         # Use fallback mock response if model fails
#         return mock_response["answer"], mock_response["reasoning"], mock_response["risk"]

# # ---------- UI ----------
# with gr.Blocks(theme=gr.themes.Soft(), css="""
# body {background-color: #f6f8fb;}
# .gr-button {border-radius: 12px; font-weight: 600;}
# .gr-textbox, .gr-dropdown {border-radius: 12px;}
# .gr-accordion {border-radius: 10px;}
# .gr-markdown {font-size: 16px;}
# """) as app:

#     # Header Section
#     gr.Markdown("## 💹 **FinWise — Your Financial Assistant**")
#     gr.Markdown("Ask your AI advisor for insights, risk evaluation, and recommendations.")
#     gr.Markdown("---")

#     # Main Input Area
#     with gr.Row():
#         user_query = gr.Textbox(
#             label="💬 Ask about your finances or investment goal",
#             placeholder="e.g., Should I invest in real estate or stocks in 2025?",
#             lines=3
#         )

#     with gr.Row():
#         query_type = gr.Dropdown(
#             label="Select Query Type",
#             choices=[
#                 "Investment Advice",
#                 "Risk Assessment",
#                 "Portfolio Optimization",
#                 "Market Analysis"
#             ],
#             value="Investment Advice"
#         )

#     with gr.Row():
#         submit_btn = gr.Button("🚀 Submit Query", variant="primary")

#     gr.Markdown("---")

#     # Output Display
#     response_panel = gr.Markdown(label="AI Response", value="")
#     with gr.Accordion("🧠 Reasoning Process", open=False):
#         reasoning_panel = gr.Markdown("")
#     with gr.Accordion("⚠️ Risk Evaluation", open=False):
#         risk_panel = gr.Markdown("")

#     gr.Markdown("---")
#     gr.Markdown("<center><small>FinWise © 2025 — Powered by Llama3-Finance Engine</small></center>")

#     # Event Binding
#     def on_submit(user_query, query_type):
#         if not user_query.strip():
#             return "❌ Please enter a valid question.", "", ""
#         answer, reasoning, risk = generate_response(user_query, query_type)
#         return answer, reasoning, risk

#     submit_btn.click(
#         on_submit,
#         inputs=[user_query, query_type],
#         outputs=[response_panel, reasoning_panel, risk_panel]
#     )

# # ---------- APP LAUNCH ----------
# if __name__ == "__main__":
#     app.launch()
