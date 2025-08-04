# table_styles.py
import streamlit as st

def render_comparison_table(comparison_df, highlight_func):
    styled_html = (
        comparison_df.style
            .apply(highlight_func, axis=1)
            .hide(axis="index")
            .to_html(index=False)
    )

    html = f"""
<style>
    table {{
        border-collapse: collapse;
        margin: 0 auto;
        font-family: Arial, sans-serif;
        width: 90%;
        max-width: 900px;
        table-layout: fixed;        /* Equal column widths */
        background-color: #1e1e1e;
        color: white;
    }}
    th, td {{
        border: 1px solid #444;
        padding: 4px 6px;
        text-align: center;         /* Center text */
        font-size: 12px;
        vertical-align: middle;
    }}
    th {{
        background-color: #333;
    }}
    tbody tr:hover {{
        background-color: #333333;
    }}
</style>
<div>
    {styled_html}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)