import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Portfolio Optimization Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, text)
        self.ln()

    def add_image(self, image_path, w=180):
        if os.path.exists(image_path):
            self.image(image_path, w=w)
            self.ln(10)

# Filter helper for clean report sections
def filter_nonzero(data_dict):
    return {k: v for k, v in data_dict.items() if v and abs(v) > 0.001}

def generate_pdf_report(weights, pie_chart_path, performance_chart_path, sentiment_scores, filters_used, output_path="Portfolio_Report.pdf", perf_summary=None, top_sector_projection=None):
    pdf = PDFReport()
    pdf.add_page()

    pdf.section_title("Summary")
    summary = "This report presents the results of a sentiment-enhanced direct indexing portfolio optimization. Stocks were filtered based on ESG scores and sector preferences."
    pdf.section_body(summary)

    weights_filtered = filter_nonzero(weights)
    sentiment_filtered = filter_nonzero(sentiment_scores)

    pdf.section_title("Optimized Portfolio Weights")
    weight_table = pd.Series(weights_filtered).sort_values(ascending=False)
    for stock, wt in weight_table.items():
        pdf.section_body(f"{stock}: {wt*100:.2f}%")

    pdf.section_title("Sentiment Scores")
    for stock, score in sentiment_filtered.items():
        pdf.section_body(f"{stock}: {score:.2f}")

    if perf_summary:
        pdf.section_title("Performance Summary")
        pdf.section_body(f"Portfolio Total Return: {perf_summary.get('portfolio_return', 0):.2f}%")
        pdf.section_body(f"SPY Benchmark Return: {perf_summary.get('spy_return', 0):.2f}%")
        pdf.section_body(f"Average Sentiment Score: {perf_summary.get('avg_sentiment', 0):.2f}")
        pdf.section_body(f"Most Bullish Stock: {perf_summary.get('top_sentiment_stock', '')} ({perf_summary.get('top_sentiment_score', 0):.2f})")

    pdf.section_title("Filters Applied")
    for label, value in filters_used.items():
        pdf.section_body(f"{label}: {value}")

    pdf.section_title("Allocation Pie Chart")
    pdf.add_image(pie_chart_path)

    pdf.section_title("Performance vs SPY")
    pdf.add_image(performance_chart_path)

    if top_sector_projection:
        pdf.section_title("What-If: All Top-Performing Sectors Included")
        pdf.section_body(f"Projected Return: {top_sector_projection.get('projected_return', 0):.2f}%")
        pdf.section_body(f"Average Sentiment: {top_sector_projection.get('avg_sentiment', 0):.2f}")
        pdf.section_body(f"Sector Impacted Portfolio Suggestion: {top_sector_projection.get('note', '')}")

    pdf.output(output_path)
    print(f"✅ PDF report generated: {output_path}")