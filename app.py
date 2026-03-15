import gradio as gr
import pandas as pd
import subprocess

def run_backtest():
    # Run the evaluation script
    result = subprocess.run(['python', 'evaluate_agents.py'], capture_output=True, text=True)
    
    # A simple parser to extract the PBO score and Leaderboard from the terminal output
    output = result.stdout
    
    pbo_status = "N/A"
    for line in output.split('\\n'):
        if "Probability of Backtest Overfitting" in line:
            pbo_status = line.strip()
            
    return pbo_status, output

with gr.Blocks(theme=gr.themes.Monochrome()) as dashboard:
    gr.Markdown("# 🤖 Quantitative AI Trading Dashboard")
    gr.Markdown("### Architecture: Meta-Labeling (DQN Direction + PPO Bet Sizing)")
    
    with gr.Row():
        run_btn = gr.Button("🚀 Execute Institutional Backtest (Out-of-Sample)", variant="primary")
        
    with gr.Row():
        pbo_display = gr.Textbox(label="Combinatorially Symmetric Cross-Validation (CSCV)", lines=1)
        
    with gr.Row():
        terminal_output = gr.Textbox(label="Evaluation Leaderboard & Terminal Logs", lines=20)
        
    run_btn.click(fn=run_backtest, outputs=[pbo_display, terminal_output])

if __name__ == "__main__":
    dashboard.launch(share=True)
