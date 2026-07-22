from train_and_inference.earnings_call_lora import executive_turns


payload = {
    "symbol": "IBM",
    "quarter": "2024Q1",
    "transcript": [
        {"speaker": "IR", "title": "Head of Investor Relations", "content": "Welcome."},
        {"speaker": "CEO", "title": "Chairman and Chief Executive Officer", "content": "Our strategy is working."},
        {"speaker": "CFO", "title": "Senior Vice President and Chief Financial Officer", "content": "Cash flow improved."},
        {"speaker": "COO", "title": "Chief Operating Officer", "content": "Operations update."},
    ],
}


rows = executive_turns(payload)
assert [row["speaker"] for row in rows] == ["CEO", "CFO"]
assert all(row["ticker"] == "IBM" and row["quarter"] == "2024Q1" for row in rows)
assert [row["text"] for row in rows] == ["Our strategy is working.", "Cash flow improved."]
print("PASS: CEO and CFO turns are retained while non-executive turns are excluded")
